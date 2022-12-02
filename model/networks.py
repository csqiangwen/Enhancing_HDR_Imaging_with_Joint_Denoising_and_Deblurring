import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import functools
from torch.optim import lr_scheduler
# from model.util import AttentionBlock
import model.restoremer as restoremer
import numpy as np
from torchvision.ops import DeformConv2d
import random
###############################################################################
# Functions
###############################################################################

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.niter_decay, gamma=0.7)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(in_channel, init_type='normal'):
    net = None

    net = Generator(in_channel=in_channel)
    
    return net

def define_D(in_channel):
    netD = None

    netD = NLayerDiscriminator(input_nc=in_channel)

    return netD

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class ResidualBlockNoBN(BaseNetwork):

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class make_dilation_dense(BaseNetwork):

    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2 + 1,
                              bias=True, dilation=2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class DRDB(BaseNetwork):

    def __init__(self, nChannels, denseLayer, growthRate):
        super(DRDB, self).__init__()
        num_channels = nChannels
        modules = []
        for i in range(denseLayer):
            modules.append(make_dilation_dense(num_channels, growthRate))
            num_channels += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(num_channels, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class Pyramid(BaseNetwork):
    def __init__(self, in_channels=6, n_feats=64):
        super(Pyramid, self).__init__()
        self.in_channels = in_channels
        self.n_feats = n_feats
        num_feat_extra = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.n_feats, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        layers = []
        for _ in range(num_feat_extra):
            layers.append(ResidualBlockNoBN())
        self.feature_extraction = nn.Sequential(*layers)

        self.downsample1 = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        x_in = self.conv1(x)
        x1 = self.feature_extraction(x_in)
        x2 = self.downsample1(x1)
        x3 = self.downsample2(x2)
        return [x1, x2, x3]


class AttentionModule(BaseNetwork):

    def __init__(self, nf=64):
        super(AttentionModule, self).__init__()

        self.attn_blks = nn.Sequential(restoremer.TransformerBlock(dim=nf, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'),
                                      restoremer.TransformerBlock(dim=nf, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'),
                                      restoremer.TransformerBlock(dim=nf, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'))
        
        self.conv_merge_0 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.conv_merge_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        [s_feats, m_feats, l_feats] = x
        hdr_feat_2 = self.attn_blks[2]([l_feats[2], s_feats[2], m_feats[2]])
        hdr_feat_2 = nn.functional.interpolate(hdr_feat_2, scale_factor=2, mode='bilinear', align_corners=False)
        hdr_feat_1 = self.attn_blks[1]([m_feats[1], s_feats[1], l_feats[1]])
        hdr_feat_1 = self.relu(self.conv_merge_1(torch.cat([hdr_feat_1, hdr_feat_2], dim=1)))
        hdr_feat_1 = nn.functional.interpolate(hdr_feat_1, scale_factor=2, mode='bilinear', align_corners=False)
        hdr_feat_0 = self.attn_blks[0]([s_feats[0], m_feats[0], l_feats[0]])
        hdr_feat = self.relu(self.conv_merge_0(torch.cat([hdr_feat_0, hdr_feat_1], dim=1)))

        return hdr_feat

    
### Attention-based Generator
class Generator(BaseNetwork):
    def __init__(self, nChannel, nDenselayer, nFeat, growthRate):
        super(Generator, self).__init__()

        self.n_channel = nChannel
        self.n_denselayer = nDenselayer
        self.n_feats = nFeat
        self.growth_rate = growthRate

        # PCD align module
        self.pyramid_feats = Pyramid(3)
        self.attn_blk = AttentionModule(nf=64)

        # feature extraction
        self.feat_exract = nn.Sequential(
            nn.Conv2d(3, nFeat, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # conv1
        self.conv1 = nn.Conv2d(self.n_feats * 4, self.n_feats, kernel_size=3, padding=1, bias=True)
        # 3 x DRDBs
        self.RDB1 = DRDB(self.n_feats, self.n_denselayer, self.growth_rate)
        self.RDB2 = DRDB(self.n_feats, self.n_denselayer, self.growth_rate)
        self.RDB3 = DRDB(self.n_feats, self.n_denselayer, self.growth_rate)
        # conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.n_feats * 3, self.n_feats, kernel_size=1, padding=0, bias=True),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, padding=1, bias=True)
        )
        # post conv
        self.post_conv = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_feats, 3, kernel_size=3, padding=1, bias=True),
            # nn.ReLU(inplace=True)
            nn.Tanh()
        )
        self.relu = nn.LeakyReLU(inplace=True)

        self.init_weights(init_type='xavier')
        
    def forward(self, x):
        [s_input, m_input, l_input] = x
        # pyramid features of linear domain
        s_feats = self.pyramid_feats(s_input)
        m_feats = self.pyramid_feats(m_input)
        l_feats = self.pyramid_feats(l_input)
        # Attention-based fusion
        coarse_hdr_feat = self.attn_blk([s_feats, m_feats, l_feats])
        # Spatial attention module
        s_feat = self.feat_exract(s_input)
        m_feat = self.feat_exract(m_input)
        l_feat = self.feat_exract(l_input)

        # fusion subnet
        F_ = torch.cat((s_feat, m_feat, l_feat, coarse_hdr_feat), 1)
        F_0 = self.conv1(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        FF = torch.cat((F_1, F_2, F_3), 1)
        FF = self.conv2(FF)
        FF = FF + coarse_hdr_feat
        res = self.post_conv(FF)
        return res
        