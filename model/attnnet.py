#-*- coding:utf-8 _*-  
""" 
@author: LiuZhen
@license: Apache Licence 
@file: adnet.py
@time: 2021/08/02
@contact: liuzhen.pwd@gmail.com
@site:  
@software: PyCharm 

"""
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import model.restoremer as restoremer


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

        self.attn_blks = nn.Sequential(restoremer.TransformerBlock_3(dim=nf, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'),
                                       restoremer.TransformerBlock_3(dim=nf, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'),
                                       restoremer.TransformerBlock_3(dim=nf, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'))
        
        self.conv_merge_0 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.conv_merge_1 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        [s_feats, m_feats, l_feats] = x
        hdr_feat_2 = self.attn_blks[2]([l_feats[2], s_feats[2], m_feats[2]])
        hdr_feat_2 = nn.functional.interpolate(hdr_feat_2, scale_factor=2, mode='bilinear', align_corners=False)
        hdr_feat_1 = self.attn_blks[1]([m_feats[1], s_feats[1], l_feats[1]])
        hdr_feat_1 = self.lrelu2(self.conv_merge_1(torch.cat([hdr_feat_1, hdr_feat_2], dim=1)))
        hdr_feat_1 = nn.functional.interpolate(hdr_feat_1, scale_factor=2, mode='bilinear', align_corners=False)
        hdr_feat_0 = self.attn_blks[0]([s_feats[0], m_feats[0], l_feats[0]])
        hdr_feat = self.lrelu1(self.conv_merge_0(torch.cat([hdr_feat_0, hdr_feat_1], dim=1)))

        return hdr_feat


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


class Downsample_block_normal(BaseNetwork):
    def __init__(self, in_channel=3, out_channel=512, need_norm=False):
        super(Downsample_block_normal, self).__init__()
        if need_norm:
            model = [nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False),
                     nn.InstanceNorm2d(out_channel),
                     nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        else:
            model = [nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=True),
                     nn.LeakyReLU(negative_slope=0.1, inplace=True)]

        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)
    
    
class Upsample_block_normal(BaseNetwork):
    def __init__(self, in_channel=3, out_channel=512, activation='relu', need_norm=False):
        super(Upsample_block_normal, self).__init__()
        if need_norm:
            model = [nn.Upsample(scale_factor=2, mode='bilinear'),
                     nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                     nn.InstanceNorm2d(out_channel)]
        else:
            model = [nn.Upsample(scale_factor=2, mode='bilinear'),
                     nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)]

        if activation == 'relu':
            model += [nn.ReLU(inplace=True)]
        elif activation == 'lrelu':
            model += [nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        elif activation == 'none':
            pass

        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)


class AttnNet(BaseNetwork):

    def __init__(self, nChannel, nDenselayer, nFeat, growthRate):
        super(AttnNet, self).__init__()
        self.n_channel = nChannel
        self.n_denselayer = nDenselayer
        self.n_feats = nFeat
        self.growth_rate = growthRate

        # PCD align module
        self.pyramid_feats = Pyramid(3)
        self.attn_blk = AttentionModule(nf=64)
        self.attn_layer = restoremer.TransformerBlock_4(dim=256, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias')
        self.side_layer = nn.Sequential(nn.Conv2d(64, 3, kernel_size=1, bias=False))

        downsample_block_hdr = []
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            ### Conv
            downsample_block_hdr.append(Downsample_block_normal(64 * mult, 64 * mult * 2))

        downsample_block_ldr = [nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
                                nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            ### Conv
            downsample_block_ldr.append(Downsample_block_normal(64 * mult, 64 * mult * 2))
    
        upsample_block = []
        for i in range(n_downsampling):  # add Upsampling layers
            mult = 2 ** (n_downsampling-i)
            if i == 1:
                upsample_block.append(Upsample_block_normal(64 * mult, 64 * mult // 2, 'none'))
            else:
                upsample_block.append(Upsample_block_normal(64 * mult, 64 * mult // 2, 'lrelu'))
        
        # post conv
        self.post_conv = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_feats, 3, kernel_size=3, padding=1, bias=True),
            nn.Tanh()
        )

        self.downsample_block_hdr = nn.Sequential(*downsample_block_hdr)
        self.downsample_block_ldr = nn.Sequential(*downsample_block_ldr)
        self.upsample_block = nn.Sequential(*upsample_block)

        self.init_weights(init_type='xavier')

    def forward(self, x):
        [s_hdr, m_hdr, l_hdr, s_ldr, m_ldr, l_ldr] = x
        # pyramid features of linear domain
        s_feats = self.pyramid_feats(s_hdr)
        m_feats = self.pyramid_feats(m_hdr)
        l_feats = self.pyramid_feats(l_hdr)
        # Attention-based fusion
        coarse_hdr_feat = self.attn_blk([s_feats, m_feats, l_feats])
        mid_res = self.side_layer(coarse_hdr_feat)

        s_feat = self.downsample_block_ldr(s_ldr)
        m_feat = self.downsample_block_ldr(m_ldr)
        l_feat = self.downsample_block_ldr(l_ldr)
        
        feat = self.downsample_block_hdr(coarse_hdr_feat)
        feat = self.attn_layer([feat, s_feat, m_feat, l_feat])
        feat = self.upsample_block[0](feat)
        feat = self.upsample_block[1](feat)
        feat = coarse_hdr_feat + feat
        res = self.post_conv(feat)

        return mid_res, res


