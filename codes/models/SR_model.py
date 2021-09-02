import logging
from collections import OrderedDict
import copy 
import sys 
import os 

import torch
from torch.multiprocessing import get_sharing_strategy
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, FSLoss, GradientLoss
from .ldp_cpt_quantize import QConv2d, my_clamp_round

logger = logging.getLogger('base')


#### for LDP 
layer_cost = []

class RunningMean(object):
    def __init__(self):
        self.reset()

    def reset(self, ratio=0.9):
        self.val = 0
        self.start = True
        self.avg = 0
        self.ratio = ratio

    def update(self, val):
        self.val = val
        if self.start:
            self.start = False
            self.avg = self.val
        else:
            self.avg = self.ratio * self.avg + (1 - self.ratio) * self.val

def get_shape(self, input, output):
    name = self.__class__.__name__

    weight_shape = self.weight.shape
    if 'onv' in name:
        try:
            layer_stride = self.stride[0]
        except:
            layer_stride = self.stride
        groups = self.groups
    input_shape = input[0].shape
    output_shape = output.shape 
    
    if 'onv' in name:
        print('{}\tweight shape {}\tinput shape {}\t output shape {}\t groups {}'.format(name, weight_shape, input_shape, output_shape, groups))
        layer_cost.append({'name': name, 'weight': weight_shape, 'input':input_shape, 'output':output_shape, 'stride':layer_stride, 'groups':groups, 'ldp': 'Q' in name})
    else:
        print('{}/tweight shape {}\tinput shape {}\t output shape {}\t'.format(name, weight_shape, input_shape, output_shape))
        layer_cost.append({'name': name, 'weight': weight_shape, 'input': input_shape, 'output': output_shape, 'ldp': 'Q' in name})

 
#def get_shape(self,input, output):
#    name = self.__class__.__name__
#    conv_shape = self.weight.shape
#    try:
#        layer_stride = self.stride[0]
#    except:
#        layer_stride = self.stride
#    input_shape = input[0].shape
#    output_shape = output.shape 
#    groups = self.groups
#
#    print('{}\tconv shape {}\tinput shape {}\t output shape {}\t groups {}'.format(name, conv_shape, input_shape, output_shape, groups))
#    layer_cost.append({'conv': conv_shape, 'input':input_shape, 'output':output_shape, 'stride':layer_stride, 'groups':groups})
#
class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        self.train_opt = train_opt 

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            self.loss_type = loss_type
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            optim_prec_params = [] 
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    if 'prec' in k:
                        optim_prec_params.append(v)
                    else:
                        optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            if train_opt['prec_opt'] == 'sgd':
                self.optimizer_P = torch.optim.SGD(optim_prec_params, lr=train_opt['prec_lr'])
            elif train_opt['prec_opt'] == 'adam':
                self.optimizer_P = torch.optim.Adam(optim_prec_params, lr=train_opt['prec_lr'], weight_decay=0, 
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

            # dynamic precision 
            self.max_bit = train_opt['max_bit']
            self.min_bit = train_opt['min_bit']
            self.bit_range = self.max_bit - self.min_bit 
            self.fix_bit = train_opt['fix_bit']
            self.grad_bit = train_opt['grad_bit']
            self.quant_lr = train_opt['quant_lr']
            
            self.init_flops_weight = train_opt['loss_flops_weight']
            self.target_flops_ratio = train_opt['target_ratio']
            self.prec_grad_reference = train_opt['reference']

            self.loss_type = train_opt['loss_type']

            self.input_size = [1, 3, 64, 64]

            self.initialize_layer_costs() 

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT
    
    def mixup_data(self, x, y, alpha=1.0, use_cuda=True): 
        '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda''' 
        batch_size = x.size()[0] 
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1 
        index = torch.randperm(batch_size).cuda() if use_cuda else torch.randperm(batch_size) 
        mixed_x = lam * x + (1 - lam) * x[index,:] 
        mixed_y = lam * y + (1 - lam) * y[index,:] 
        return mixed_x, mixed_y
    
    def optimize_parameters(self, curr_step, tb_logger, total_iters=None):
        # tb_logger: tensorboard logger 

        '''add mixup operation'''
#         self.var_L, self.real_H = self.mixup_data(self.var_L, self.real_H)
        
        self.fake_H = self.netG(self.var_L, self.grad_bit)
        if self.loss_type == 'fs':
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H) + self.l_fs_w * self.cri_fs(self.fake_H, self.real_H)
        elif self.loss_type == 'grad':
            l1 = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
            lg = self.l_grad_w * self.gradloss(self.fake_H, self.real_H)
            l_pix = l1 + lg
        elif self.loss_type == 'grad_fs':
            l1 = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
            lg = self.l_grad_w * self.gradloss(self.fake_H, self.real_H)
            lfs = self.l_fs_w * self.cri_fs(self.fake_H, self.real_H)
            l_pix = l1 + lg + lfs
        else:
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        
        if self.train_opt['DP']:
            # if self.train_opt['change']: 
            #     if curr_step > (total_iters*0.75):
            #         self.train_opt['loss_flops_weight'] = self.init_flops_weight * 0.25 
            #     elif curr_step > (total_iters * 0.5): 
            #         self.train_opt['loss_flops_weight'] = self.init_flops_weight * 0.5 
            #     else:
            #         self.train_opt['loss_flops_weight'] = self.init_flops_weight 
            cost_ratio, prec_grad_mean = self.get_efficiency_loss(self.train_opt['loss_flops_weight'], self.train_opt['loss_flops_weight'])  
            # flops_weight = l_pix.item() / (flops_loss.item() + 1e-8)
            # l_pix += flops_weight * flops_loss * self.train_opt['loss_flops_weight']
        
        l_pix.backward()

        nn.utils.clip_grad_norm_(self.netG.parameters(), 5) 
        
        self.optimizer_G.step()

        if self.train_opt['DP']:
            self.optimizer_P.step()
            self.optimizer_P.zero_grad() 

            # tb logging related 
            if curr_step % self.train_opt['tb_logging_interval'] == 0:
                prec_list = self.tb_info_logging(tb_logger, curr_step) # TODO: log with name of layer, includes prec, grad, bit_grad, returns prec_list 
            else:
                prec_list = None
                
        self.optimizer_G.zero_grad()
        

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        if self.loss_type == 'grad':
            self.log_dict['l_1'] = l1.item()
            self.log_dict['l_grad'] = lg.item()
        if self.loss_type == 'grad_fs':
            self.log_dict['l_1'] = l1.item()
            self.log_dict['l_grad'] = lg.item()
            self.log_dict['l_fs'] = lfs.item()
        
        if self.train_opt['DP']:
            return prec_list, cost_ratio 
        else:
            return None, 1.0 

    def get_efficiency_loss(self, flops_weight=None, grad_penalty_val=None):
        with torch.no_grad():
            curr_flops = self.total_static_flops 
            cnt = 0 
            for la in self.netG.modules():
                if isinstance(la, QConv2d):
                    layer_prec = torch.clamp(torch.round(la.prec_w * self.bit_range + self.min_bit), self.min_bit, self.max_bit).item() 
                    curr_flops += self.ldp_layer_flops[cnt] * (layer_prec * layer_prec / self.max_bit / self.max_bit + 2 * layer_prec / self.max_bit) / 3
                    cnt += 1 
            curr_ratio = curr_flops / self.total_flops 

            if curr_flops > self.target_flops:
                prec_grad_list = []
                for la in self.netG.modules():
                    if isinstance(la, QConv2d):
                        prec_grad_list.append(la.prec_w.grad.item())
                prec_grad_list = torch.tensor(prec_grad_list)
                max_grad = torch.max(torch.abs(prec_grad_list))
                self.prec_grad_running_mean.update(torch.mean(torch.abs(prec_grad_list)))

                assert flops_weight is not None 
                # reduce based on prec split ratio 
                actual_grad = self.layer_cost_ratio * self.bit_range
                if self.prec_grad_reference == 'static':
                    actual_grad_scale = flops_weight 
                elif self.prec_grad_reference == 'prec_grad':
                    actual_grad_scale = flops_weight * self.prec_grad_running_mean.avg / np.mean(actual_grad)
                else:
                    raise NotImplementedError 
                
                actual_grad = torch.tensor(actual_grad) * actual_grad_scale
                idx = 0 
                for la in self.netG.modules():
                    if isinstance(la, QConv2d):
                        la.prec_w.grad = torch.tensor(la.prec_w.grad + actual_grad[idx])
                        idx += 1 

        return curr_ratio, self.prec_grad_running_mean.avg

    def set_true_grad(self, true_grad, reference):
        self.true_grad = true_grad 
        self.prec_grad_running_mean = RunningMean()
        self.prec_grad_reference = reference  
        if self.true_grad == 'actual_grad':
            # get layerwise prec cost
            self.layer_cost_ratio = [lf / self.total_flops for lf in self.ldp_layer_flops] 

    def inference_cost(self, prec_list):
        flops = self.total_static_flops
        ldp_flops = self.ldp_layer_flops * prec_list * prec_list / self.max_bit / self.max_bit 
        ldp_flops = sum(ldp_flops)
        flops += ldp_flops 
        infer_cost = flops / self.total_flops
        return infer_cost 

    def initialize_layer_costs(self):
        tmp_self = copy.deepcopy(self)
        
        self.ldp_layer_flops = []
        self.static_layer_flops = []  
        
        for layer in tmp_self.netG.modules():
            if isinstance(layer, nn.Conv2d):
                # extract convolution layers 
                layer.register_forward_hook(get_shape)
        test_input = torch.rand(self.input_size)
        _ = tmp_self.netG(test_input, )
        self.layer_shapes = layer_cost
        del tmp_self

        for l in range(len(self.layer_shapes)):
            curr_stat = self.layer_shapes[l]
            name = curr_stat['name']
            ldp = curr_stat['ldp']
            if 'onv' in name:
                st = curr_stat['stride']
                inp = curr_stat['input']
            kernel = curr_stat['weight']
            if 'onv' in name:
                flops = (kernel[1] * kernel[2] * kernel[3] + 1) * kernel[0] * inp[2] * inp[3] / st / st
            else: 
                flops = (2 * kernel[0] - 1) * kernel[1]

            flops = flops * self.max_bit * self.max_bit / 32/ 32  # initial flops count with 8-bit
            if ldp:
                self.ldp_layer_flops.append(flops)
            else:
                self.static_layer_flops.append(flops) 
        
        self.total_flops = sum(self.ldp_layer_flops) + sum(self.static_layer_flops) # only forward flops is considered 
        self.total_static_flops = sum(self.static_layer_flops)
        self.total_ldp_flops = sum(self.ldp_layer_flops)
        self.ldp_flops_ratio = self.total_ldp_flops / self.total_flops

        print('total flops is {}\ntotal ldp flops is {}\ttotal static flops is {}\nldp flops ratio is {:.3f}'.format(self.total_flops, 
                            self.total_ldp_flops, self.total_static_flops, self.ldp_flops_ratio))
        
        self.target_flops = self.total_flops * self.target_flops_ratio 

        self.set_true_grad('actual_grad', self.prec_grad_reference)
    
    def reset_prec(self):
        for la in self.netG.modules():
            if isinstance(la, QConv2d):
                if la.prec_w > 1.1:
                    nn.init.ones_(la.prec_w)
                elif la.prec_w < -0.1:
                    nn.init.zeros_(la.prec_w)

                    
    def tb_info_logging(self, tb_logger, curr_step):
        prec_list = [] 
        # TODO: add layer weight gradient into this function 
        for name, param in self.netG.module.named_parameters():
            if 'prec' in name:
                prec = np.clip(np.round(param.item() * (self.max_bit - self.min_bit) + self.min_bit), self.min_bit, self.max_bit)
                prec_list.append(prec) 
                if curr_step %  self.train_opt['tb_logging_interval'] == 0:
                    tb_logger.add_scalar('{}prec/iter'.format(name), prec, curr_step)

        return prec_list 

    def test(self):
        self.netG.eval()
        
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
    
#     def load(self):
#         load_path_G_1 = self.opt['path']['pretrain_model_G_1']
#         load_path_G_2 = self.opt['path']['pretrain_model_G_2']
#         load_path_Gs=[load_path_G_1, load_path_G_2]
        
#         load_path_G = self.opt['path']['pretrain_model_G']
#         if load_path_G is not None:
#             logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
#             self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
#         if load_path_G_1 is not None:
#             logger.info('Loading model for 3net [{:s}] ...'.format(load_path_G_1))
#             logger.info('Loading model for 3net [{:s}] ...'.format(load_path_G_2))
#             self.load_network_part(load_path_Gs, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
