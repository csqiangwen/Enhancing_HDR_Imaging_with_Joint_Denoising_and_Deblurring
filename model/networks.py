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


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetBlock(BaseNetwork):
    def __init__(self, dim, padding_type, norm_layer, groups=1, activation=nn.ReLU(True), use_dropout=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, groups, activation, use_dropout)
        self.init_weights(init_type='xavier')

    def build_conv_block(self, dim, padding_type, norm_layer, groups, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False, groups=groups),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, groups=groups)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Downsample_block_normal(BaseNetwork):
    def __init__(self, in_channel=3, out_channel=512):
        super(Downsample_block_normal, self).__init__()
        # model = [nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2,
        #                       padding=1, bias=False),
        #          nn.InstanceNorm2d(out_channel),
        #          nn.ReLU(True)]

        model = [nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
                 nn.PixelUnshuffle(2),
                 nn.ReLU(True)]

        self.model = nn.Sequential(*model)
        self.init_weights(init_type='xavier')
        
    def forward(self, x):
        return self.model(x)
    
    
class Upsample_block_normal(BaseNetwork):
    def __init__(self, in_channel=3, out_channel=512):
        super(Upsample_block_normal, self).__init__()
        # model = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #          nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1,
        #                       padding=1, bias=False),
        #          nn.InstanceNorm2d(out_channel),
        #          nn.ReLU(True)]

        model = [nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
                 nn.PixelShuffle(2),
                 nn.ReLU(True)]

        self.model = nn.Sequential(*model)
        self.init_weights(init_type='xavier')
        
    def forward(self, x):
        return self.model(x)


class Warm_up_block(BaseNetwork):
    def __init__(self, in_channel=3, out_channel=512, activation='relu'):
        super(Warm_up_block, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=0, bias=True)]

        if activation == 'relu':
            model += [nn.ReLU(inplace=True)]
        elif activation == 'tanh':
            model += [nn.Tanh()]
        elif activation == 'none':
            pass
        self.model = nn.Sequential(*model)
        self.init_weights(init_type='xavier')
        
    def forward(self, x):
        return self.model(x)

    
### Attention-based Generator
class Generator(BaseNetwork):
    def __init__(self, in_channel=3, out_channel=3, ngf=64, n_downsampling=2, res_block_num=4):
        super(Generator, self).__init__()
                
        main_branch = [Warm_up_block(in_channel+2, ngf, activation='relu')]
        ref_branch = [Warm_up_block(in_channel, ngf, activation='relu')]

        for i in range(n_downsampling):
            mult = 2 ** i
            main_branch.append(Downsample_block_normal(ngf * mult, ngf * mult // 2))
            ref_branch.append(Downsample_block_normal(ngf * mult, ngf * mult // 2))

        # for i in range(res_block_num//2):
        #     main_branch.append(ResnetBlock(ngf * mult * 2, 'reflect', nn.InstanceNorm2d, groups=1, use_dropout=True))
        #     ref_branch.append(ResnetBlock(ngf * mult * 2, 'reflect', nn.InstanceNorm2d, groups=1, use_dropout=True))

        restoremers = []
        for i in range(n_downsampling+1):
            mult = 2 ** i
            restoremers.append(restoremer.TransformerBlock(dim=ngf * mult, num_heads=1, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'))
        
        upsample_block = []
        for i in range(n_downsampling):  # add Upsampling layers
            mult = 2 ** (n_downsampling-i)
            if i == 0:
                upsample_block.append(Upsample_block_normal(ngf * mult, ngf * mult * 2))
            else:
                upsample_block.append(Upsample_block_normal(ngf * mult * 2, ngf * mult * 2))

        upsample_block += [Warm_up_block(ngf * mult, out_channel, activation='tanh')]

        side_branch = [nn.Conv2d(256, 3, kernel_size=1, padding=0, bias=True),
                       nn.Tanh()]
            
        self.main_branch = nn.Sequential(*main_branch)
        self.ref_branch = nn.Sequential(*ref_branch)
        self.restoremers = nn.Sequential(*restoremers)
        self.upsample_block = nn.Sequential(*upsample_block)
        self.side_branch = nn.Sequential(*side_branch)

        self.init_weights(init_type='xavier')
        
    def forward(self, x):
        [s_LHDR, m_LHDR, l_LHDR] = x
        b, c, h, w = s_LHDR.shape
        m_feats4cat = []
        s_feat = s_LHDR
        m_feat = m_LHDR
        l_feat = l_LHDR
        for i, (main_layer, ref_layer) in enumerate(zip(self.main_branch, self.ref_branch)):
            s_feat = ref_layer(s_feat)
            m_feat = main_layer(m_feat)
            l_feat = ref_layer(l_feat)
            m_feats4cat.append(self.restoremers[i]([s_feat, m_feat, l_feat]))
        _, c, sh, sw = m_feat.shape
        multi_feats = torch.stack([s_feat, m_feats4cat[-1], l_feat], dim=1)
        small_output = self.side_branch(multi_feats.view(-1, *multi_feats.shape[2:])).view(b, 3, 3, sh, sw)
        hdr_feat = m_feat
        feats4cat_n = len(m_feats4cat)
        for i, img_layer in enumerate(self.upsample_block):
            if i == 0:
                hdr_feat = img_layer(m_feats4cat[-1])
            else:
                hdr_feat = img_layer(torch.cat([hdr_feat, m_feats4cat[feats4cat_n-i-1]], dim=1))
        output = hdr_feat
        small_output_list = torch.chunk(small_output, 3, dim=1)

        return output, small_output_list


class NLayerDiscriminator(BaseNetwork):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=4, norm_layer=nn.InstanceNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

        self.init_weights(init_type='xavier')

    def forward(self, input):
        """Standard forward."""
        return self.model(input)