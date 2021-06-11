import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util
import models.quantize as Q 

Conv2d = Q.QConv2d
Linear = Q.QLinear

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out
    
class PAConv(nn.Module):

    def __init__(self, nf, k_size=3, max_bit=None, min_bit=None):

        super(PAConv, self).__init__()
        self.k2 = Conv2d(nf, nf, 1, max_bit=max_bit, min_bit=min_bit) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False, max_bit=max_bit, min_bit=min_bit) # 3x3 convolution
        self.k4 = Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False, max_bit=max_bit, min_bit=min_bit) # 3x3 convolution

    def forward(self, x, fix_bit=None):

        y = self.k2(x, fix_bit=fix_bit)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x, fix_bit=fix_bit), y)
        out = self.k4(out, fix_bit=fix_bit)

        return out
        
class SCPA(nn.Module):
    
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """

    def __init__(self, nf, reduction=2, stride=1, dilation=1, max_bit=None, min_bit=None):
        super(SCPA, self).__init__()
        group_width = nf // reduction
        
        self.conv1_a = Conv2d(nf, group_width, kernel_size=1, bias=False, max_bit=max_bit, min_bit=min_bit)
        self.conv1_b = Conv2d(nf, group_width, kernel_size=1, bias=False, max_bit=max_bit, min_bit=min_bit)
        
        self.k1 = Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        bias=False, max_bit=max_bit, min_bit=min_bit)
        
        self.PAConv = PAConv(group_width, max_bit=max_bit, min_bit=min_bit)
        
        self.conv3 = Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False, max_bit=max_bit, min_bit=min_bit)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, fix_bit=None):
        residual = x

        out_a= self.conv1_a(x, fix_bit=fix_bit)
        out_b = self.conv1_b(x, fix_bit=fix_bit)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a, fix_bit=fix_bit)
        out_b = self.PAConv(out_b, fix_bit=fix_bit)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1), fix_bit=fix_bit)
        out += residual

        return out
    
class PAN(nn.Module):
    
    def __init__(self, in_nc, out_nc, nf, unf, nb, scale=4, max_bit=None, min_bit=None):
        super(PAN, self).__init__()
        # SCPA
        SCPA_block_f = functools.partial(SCPA, nf=nf, reduction=2, max_bit=max_bit, min_bit=min_bit)
        self.scale = scale
        
        ### first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        ### main blocks
        self.SCPA_trunk = arch_util.make_layer(SCPA_block_f, nb)
        self.trunk_conv = Conv2d(nf, nf, 3, 1, 1, bias=True, max_bit=max_bit, min_bit=min_bit)
        
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True)
        self.att1 = PA(unf)
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
        
        if self.scale == 4:
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            self.att2 = PA(unf)
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True)
            
        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, fix_bit=None):
        
        # first conv
        fea = self.conv_first(x)
        
        # main blocks 
        trunk = fea 
        for m in self.SCPA_trunk:
            trunk = m(trunk, fix_bit=fix_bit)
        trunk = self.trunk_conv(trunk, fix_bit=fix_bit)
        fea = fea + trunk
        
        # upsampling 
        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))
        
        out = self.conv_last(fea)
        
        ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + ILR
        return out
 
