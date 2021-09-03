from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)


def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, reduce_type='mean', keepdim=False,
                      true_zero=False):
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        if x_flat.dim() == 1:
            min_values = _deflatten_as(x_flat.min(), x)
            max_values = _deflatten_as(x_flat.max(), x)
        else:
            min_values = _deflatten_as(x_flat.min(-1)[0], x)
            max_values = _deflatten_as(x_flat.max(-1)[0], x)

        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]

        range_values = max_values - min_values
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)

class my_clamp_round(InplaceFunction):

    @staticmethod
    def forward(ctx, input, min_value, max_value):
        ctx.input = input
        ctx.min = min_value
        ctx.max = max_value
        return torch.clamp(torch.round(input), min_value, max_value)

    @staticmethod
    def backward(ctx, grad_output):
        # original impl, no backprop with out of bit range 
#        grad_input = grad_output.clone()
#        mask = (ctx.input > ctx.min) * (ctx.input < ctx.max)
#        grad_input = mask.float() * grad_input

        # my impl, backprop for all range
        grad_input = grad_output.clone() 
        return grad_input, None, None


class FakeWeight(InplaceFunction):
    @staticmethod 
    def forward(ctx, input): 
        
        return input
    
    @staticmethod 
    def backward(ctx, output):
        input = output.clone()
        return input 

class UniformQuantize():

    @staticmethod
    def uni_quantize(input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False,
                prec_sf=None, min_bit=None, max_bit=None):

        inplace = inplace

        output = input.clone()

        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                output, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim)

        zero_point = qparams.zero_point

        if prec_sf is not None:
            # print('running with learnable prec')
            # original impl 
            # prec = my_clamp_round().apply(prec_sf*num_bits, min_bit, max_bit)
            # my impl. prec_sf range from 0-1 0 --> min_bit 1 --> max_bit 
            bit_range = max_bit - min_bit
            num_bits = my_clamp_round().apply(prec_sf * bit_range + min_bit, min_bit, max_bit)
        else:
            num_bits = qparams.num_bits

        qmin = -(2. ** (num_bits - 1)) if signed else 0.
        qmax = qmin + 2. ** num_bits - 1.
        scale = qparams.range / (qmax - qmin)
        if scale.is_cuda:
            min_scale = torch.tensor(1e-8).expand_as(scale).cuda()
        else:
            min_scale = torch.tensor(1e-8).expand_as(scale)
        scale = torch.max(scale, min_scale)

        output.add_(qmin * scale - zero_point).div_(scale)
        if stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)
        # quantize
        output = my_clamp_round().apply(output, qmin, int(qmax))

        if dequantize:
            output.mul_(scale).add_(
                zero_point - qmin * scale)  # dequantize
        return output


class UniformQuantizeGrad(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD,
                reduce_dim=0, dequantize=True, signed=False, stochastic=True):
        ctx.num_bits = num_bits
        ctx.qparams = qparams
        ctx.flatten_dims = flatten_dims
        ctx.stochastic = stochastic
        ctx.signed = signed
        ctx.dequantize = dequantize
        ctx.reduce_dim = reduce_dim
        ctx.inplace = False
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        qparams = ctx.qparams
        with torch.no_grad():
            if qparams is None:
                assert ctx.num_bits is not None, "either provide qparams of num_bits to quantize"
                qparams = calculate_qparams(
                    grad_output, num_bits=ctx.num_bits, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                    reduce_type='extreme')

            grad_input = quantize(grad_output, num_bits=None,
                                  qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                                  dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
        return grad_input, None, None, None, None, None, None, None


def quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False,
             stochastic=False, inplace=False, 
             prec_sf=None, max_bit=None, min_bit=None):
    return UniformQuantize.uni_quantize(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic,
                                       inplace, prec_sf=prec_sf, max_bit=max_bit, min_bit=min_bit)

    return x


def quantize_grad(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0, dequantize=True,
                  signed=False, stochastic=True):
    if qparams:
        if qparams.num_bits:
            return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed,
                                               stochastic)
    elif num_bits:
        return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed,
                                           stochastic)

    return x

def fake_weight(x):
    return FakeWeight().apply(x)


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN,
                 inplace=False, dequantize=True, stochastic=False, momentum=0.9, measure=False):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_zero_point', torch.zeros(*shape_measure))
        self.register_buffer('running_range', torch.zeros(*shape_measure))
        self.measure = measure
        if self.measure:
            self.register_buffer('num_measured', torch.zeros(1))
        self.flatten_dims = flatten_dims
        self.momentum = momentum
        self.dequantize = dequantize
        self.stochastic = stochastic
        self.inplace = inplace

    def forward(self, input, num_bits, qparams=None, prec_sf=None, max_bit=None, min_bit=None):

        if self.training or self.measure:
            if qparams is None:
                qparams = calculate_qparams(
                    input, num_bits=num_bits, flatten_dims=self.flatten_dims, reduce_dim=0, reduce_type='extreme')
            with torch.no_grad():
                if self.measure:
                    momentum = self.num_measured / (self.num_measured + 1)
                    self.num_measured += 1
                else:
                    momentum = self.momentum
                self.running_zero_point.mul_(momentum).add_(
                    qparams.zero_point * (1 - momentum))
                self.running_range.mul_(momentum).add_(
                    qparams.range * (1 - momentum))
        else:
            qparams = QParams(range=self.running_range,
                              zero_point=self.running_zero_point, num_bits=num_bits)
        if self.measure:
            return input
        else:
            q_input = quantize(input, qparams=qparams, dequantize=self.dequantize,
                               stochastic=self.stochastic, inplace=self.inplace, prec_sf=prec_sf, min_bit=min_bit, max_bit=max_bit)
            return q_input


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 num_bits=8, max_bit=8, min_bit=3):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)

        self.quantize_input = QuantMeasure(shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
        self.stride = stride

        self.prec_w = nn.Parameter(torch.tensor(0.0))
        self.max_bit = max_bit 
        self.min_bit = min_bit 
        self.bit_range = max_bit - min_bit 
        self.num_bits = num_bits 

#    def forward(self, input, num_bits, num_grad_bits):
    def forward(self, input, fix_bit=None, grad_bit=16):
        num_grad_bits = grad_bit
        num_bits = self.num_bits 
        if fix_bit is not None:
            num_bits = fix_bit 
            if num_bits == 0:
                output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
                return output

            if self.bias is not None:
#                 qbias = quantize(
#                     self.bias, num_bits=num_bits,
#                     flatten_dims=(0, -1))
                qbias = quantize(self.bias.detach(), num_bits=num_bits, 
                    flatten_dims=(0, -1)) 
                qbias = qbias - fake_weight(self.bias).detach() + fake_weight(self.bias) 

            else:
                qbias = None

            weight_qparams = calculate_qparams(self.weight, num_bits=num_bits, flatten_dims=(1, -1),
                                            reduce_dim=None)
#            qweight = quantize(self.weight, qparams=weight_qparams)
            qweight = quantize(self.weight.detach(), qparams=weight_qparams)
            qweight = qweight - fake_weight(self.weight).detach() + fake_weight(self.weight) 

            qinput = self.quantize_input(input, num_bits)
            output = F.conv2d(qinput, qweight, qbias, self.stride, self.padding, self.dilation, self.groups)
            output = quantize_grad(output, num_bits=num_grad_bits, flatten_dims=(1, -1))

        else:
            if self.bias is not None: 
                # qbias = quantize(
                #     self.bias, num_bits=num_bits,
                #     flatten_dims=(0, -1))
                qbias = quantize(self.bias.detach(), num_bits=num_bits, 
                    flatten_dims=(0, -1), prec_sf=self.prec_w, max_bit=self.max_bit, min_bit=self.min_bit) 
                qbias = qbias - fake_weight(self.bias).detach() + fake_weight(self.bias) 
            else:
                qbias = None 
            weight_qparams = calculate_qparams(self.weight, num_bits=num_bits, flatten_dims=(1, -1),
                                            reduce_dim=None)
            qweight = quantize(self.weight.detach(), qparams=weight_qparams, prec_sf=self.prec_w, max_bit=self.max_bit, min_bit=self.min_bit)
            qweight = qweight - fake_weight(self.weight).detach() + fake_weight(self.weight) 

            qinput = self.quantize_input(input, num_bits, prec_sf=self.prec_w, max_bit=self.max_bit, min_bit=self.min_bit)
            output = F.conv2d(qinput, qweight, qbias, self.stride, self.padding, self.dilation, self.groups)
            output = quantize_grad(output, num_bits=num_grad_bits, flatten_dims=(1, -1))
        return output


if __name__ == '__main__':
    x = torch.rand(2, 3)
    x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
    print(x)
    print(x_q)
