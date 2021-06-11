import logging
from collections import OrderedDict
import copy 

import torch
from torch.multiprocessing import get_sharing_strategy
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import numpy as np
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, FSLoss, GradientLoss
from .quantize import QConv2d, my_clamp_round

logger = logging.getLogger('base')


#### for LDP 
layer_cost = []

def get_shape(self,input, output):
    name = self.__class__.__name__
    conv_shape = self.weight.shape
    layer_stride = self.stride[0]
    input_shape = input[0].shape
    output_shape = output.shape 
    groups = self.groups

    print('{}\tconv shape {}\tinput shape {}\t output shape {}\t groups {}'.format(name, conv_shape, input_shape, output_shape, groups))
    layer_cost.append({'conv': conv_shape, 'input':input_shape, 'output':output_shape, 'stride':layer_stride, 'groups':groups})

class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        self.opt = train_opt 

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

            self.init_flops_weight = train_opt['loss_flops_weight']
            self.target_flops_ratio = train_opt['target_ratio']

            self.loss_type = train_opt['loss_type']

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
    
    def optimize_parameters(self, curr_step, tb_logger, total_iters=None, fix_bit=None):
        # tb_logger: tensorboard logger 

        '''add mixup operation'''
#         self.var_L, self.real_H = self.mixup_data(self.var_L, self.real_H)
        
        self.fake_H = self.netG(self.var_L, fix_bit=fix_bit)
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
        
        if self.opt['DP']:
            if self.opt['change']: 
                if curr_step > (total_iters*0.75):
                    self.opt['loss_flops_weight'] = self.init_flops_weight * 0.25 
                elif curr_step > (total_iters * 0.5): 
                    self.opt['loss_flops_weight'] = self.init_flops_weight * 0.5 
                else:
                    self.opt['loss_flops_weight'] = self.init_flops_weight 
            flops_loss, total_flops = self.get_effieiency_loss(self.opt['target_ratio'], self.opt['loss_type']) # TDOO: implement this 
            flops_weight = l_pix.item() / (flops_loss.item() + 1e-8)
            l_pix += flops_weight * flops_loss * self.opt['loss_flops_weight']
        
        l_pix.backward()
        self.optimizer_G.step()

        if self.opt['DP']:
            self.optimizer_P.step()
            self.optimizer_P.zero_grad() 

            # tb logging related 
            if curr_step % self.opt['tb_logging_interval'] == 0:
                prec_list = self.tb_info_logging(tb_logger, curr_step) # TODO: log with name of layer, includes prec, grad, bit_grad, returns prec_list 
                
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
        
        if self.opt['DP']:
            return prec_list, total_flops 
        else:
            return None, None 

    def get_efficiency_loss(self):
        bp = self.opt['calc_bp_cost']
        assert self.loss_type in ['thres']
        if bp == True:
            curr_flops = self.total_fix_flops * 3 
        else: 
            curr_flops = self.total_fix_flops 
        cnt = 0 
        for layer in self.netG.modules():
            if isinstance(layer, QConv2d):
                layer_prec = my_clamp_round().apply(layer.prec_w * self.bit_range + self.min_bit, self.min_bit, self.max_bit)
                if bp == True:
                    # count bp cost in metric 
                    curr_flops += self.ldp_layer_cost[cnt] * layer_prec * layer_prec / 8 / 8 + 2 * self.ldp_layer_cost[cnt] * layer_prec / 8 
                elif bp == False:
                    curr_flops += self.ldp_layer_cost[cnt] * layer_prec * layer_prec / 8 / 8 
        if bp == True:
            curr_flops = curr_flops / 3 
        if self.loss_type == 'thres':
            if curr_flops > self.target_flops:
                return curr_flops, curr_flops
            else:
                return torch.tensor(0.0), curr_flops 

    def initialize_layer_costs(self):
        tmp_model = copy.deepcopy(self.netG)
        for l in tmp_model.modules():
            if isinstance(l, nn.Conv2d):
                l.register_forward_hook(get_shape)
        test_input = torch.rand(1, 3, 224, 224)
        _ = tmp_model(test_input)
        self.layer_shape = layer_cost 

        del tmp_model

        cnt = 0 
        self.fix_layer_cost = []
        self.ldp_layer_cost = [] 

        for la in self.netG.modules():
            if isinstance(la, nn.Conv2d):
                # calculate layer cost 
                curr_stat = self.layer_shape[cnt]
                st = curr_stat['stride']
                inp = curr_stat['input']
                kernel = curr_stat['conv']
                flops = (kernel[0] * kernel[2] * kernel[3] + 1) * kernel[1] * inp[2] * inp[3] / st / st 
                flops = flops * 8 * 8 / 32 / 32 
                if hasattr(la, 'prec_w'):
                    self.ldp_layer_cost.append(flops)
                else:
                    self.fix_layer_cost.append(flops) 
                cnt += 1 
        self.total_ldp_flops = np.sum(self.ldp_layer_cost)
        self.total_fix_flops = np.sum(self.fix_layer_cost) 
        self.total_flops = self.total_ldp_flops + self.total_fix_flops 
        
        self.target_flops = self.total_flops * self.target_flops_ratio

    def tb_info_logging(self, tb_logger, curr_step):
        prec_list = [] 
        # TODO: add layer weight gradient into this function 
        for name, param in self.netG.module.named_parameters():
            if 'prec' in name:
                prec = np.clip(np.round(param.item() * (self.max_bit - self.min_bit) + self.min_bit), self.min_bit, self.max_bit)
                prec_list.append(prec) 
                tb_logger.add_scalar('{}prec/iter'.format(name), prec, curr_step)
                bit_grad = param.grad 
                tb_logger.add_scalar('{}bitgrad/iter'.format(name), bit_grad, curr_step) 
        return prec_list 

    def test(self, fix_bit=None):
        self.netG.eval()
        
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L, fix_bit=fix_bit)
        self.netG.train()

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)
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
