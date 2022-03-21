from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data.dataset import T
import util.common_util as image_util
import util.hdr_util as hdr_util
from .base_model import BaseModel
from torch.nn.parallel import DataParallel
from . import networks
import model.util as model_util
from .vgg16_loss import VGGLoss
from model.eval_tools import compute_psnr
import torch.nn.functional as F
import cv2
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
import lpips
import os

# loss_fn_alex = lpips.LPIPS(net='alex').cuda()

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, datatype='image'):
    if datatype == 'image':
        img = img / 2 + 0.5     # unnormalize
        npimg = img.cpu().numpy()
    else:
        img = 1-img
        npimg = img.cpu().numpy()
    
    # return np.transpose(npimg, (1, 2, 0))
    return npimg
        
class HDRNet(BaseModel):
    def name(self):
        return 'HDRNet'

    def initialize(self, opt):
        self.opt = opt
        BaseModel.initialize(self, opt)

        if self.isTrain:
            # For tensorboardX
            self.writer = SummaryWriter()
            # load/define networks
            self.netG = networks.define_G(in_channel=6).to(self.device)
            self.netD = networks.define_D(in_channel=3).to(self.device)

            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionGAN = networks.GANLoss(gan_mode='lsgan').to(self.device)
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionTV = model_util.TVLoss()
            self.criterionVGG = VGGLoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.networks = []
            self.optimizers = []
            self.schedulers = []
            self.networks.append(self.netG)
            self.networks.append(self.netD)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))
                
            if opt.continue_train:
                which_iter = opt.which_iter
                self.load_states(self.netG, self.optimizer_G, self.schedulers[0], 'G', which_iter)
                self.load_states(self.netD, self.optimizer_D, self.schedulers[1], 'D', which_iter)
                
            if len(self.gpu_ids) > 0:
                self.netG = DataParallel(self.netG)
                self.netD = DataParallel(self.netD)
                
            self.netG.train()
            self.netD.train()
        
        if not self.isTrain:
            # load/define networks
            self.netG = networks.define_G(in_channel=6)

            which_iter = opt.which_iter
            self.load_states_simple(self.netG, 'G', which_iter)
            
            self.netG.eval()
            self.netG = self.netG.cuda()
            
            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            print('-----------------------------------------------')


    def img_train_forward(self, data):
        self.img_sample = {}
        self.img_sample['s_HDR'] = data['s_HDR'].to(self.device)
        self.img_sample['m_HDR'] = data['m_HDR'].to(self.device)
        self.img_sample['l_HDR'] = data['l_HDR'].to(self.device)
        self.img_sample['GT_HDR'] = data['GT_HDR'].to(self.device)

        self.img_sample['bright_c'] = data['bright_c'].to(self.device)
        self.img_sample['dark_c'] = data['dark_c'].to(self.device)  
    
        self.img_sample['s_LDR'] = data['s_LDR'].to(self.device)
        self.img_sample['m_LDR'] = data['m_LDR'].to(self.device)
        self.img_sample['l_LDR'] = data['l_LDR'].to(self.device)


    def D_optim(self, fake, real):
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        real_feat = self.netD(real)
        fake_feat = self.netD(fake.detach())
        dis_real_loss = self.criterionGAN(real_feat, True)
        dis_fake_loss = self.criterionGAN(fake_feat, False)
        DLoss = (dis_real_loss + dis_fake_loss) / 2
        DLoss.backward()
        torch.nn.utils.clip_grad_norm(self.netD.parameters(), 10)
        self.optimizer_D.step()
        self.DLoss = DLoss.item()
        return


    def img_optimize_parameters(self, iteration):

        ### input shape [B, C, H, W]
        B, C, H, W = self.img_sample['s_HDR'].shape

        self.optimizer_G.zero_grad()
        
        fake_HDR, small_outputs = self.netG([torch.cat([self.img_sample['s_LDR'], self.img_sample['s_HDR']], dim=1),
                                             torch.cat([self.img_sample['m_LDR'], self.img_sample['m_HDR'],
                                                        self.img_sample['bright_c'], self.img_sample['dark_c']], dim=1),
                                             torch.cat([self.img_sample['l_LDR'], self.img_sample['l_HDR']], dim=1)])

        fake_HDR_tm = hdr_util.tonemap(fake_HDR)
        GT_HDR_tm = hdr_util.tonemap(self.img_sample['GT_HDR'])
        
        # self.D_optim(fake_HDR_tm, GT_HDR_tm)
        # self.set_requires_grad([self.netD], False)
        # Loss_GAN = self.criterionGAN(self.netD(fake_HDR_tm), True)

        Loss_L1 = self.criterionL1(fake_HDR, self.img_sample['GT_HDR']) * 10

        Loss_GAN = Loss_L1 * 0
        self.DLoss = Loss_L1 * 0

        Loss_VGG_tm = self.criterionVGG(fake_HDR_tm, GT_HDR_tm) * 10
        
        Loss_small = (self.criterionL2(small_outputs[0][:, 0], F.interpolate(self.img_sample['s_HDR'], scale_factor=0.25)) +\
                      self.criterionL2(small_outputs[1][:, 0], F.interpolate(self.img_sample['GT_HDR'], scale_factor=0.25)) +\
                      self.criterionL2(small_outputs[2][:, 0], F.interpolate(self.img_sample['l_HDR'], scale_factor=0.25))) * 5
        
        whole_loss = Loss_L1 + Loss_VGG_tm + Loss_small# + Loss_GAN

        self.writer.add_scalar("Image/Loss_GAN", Loss_GAN, iteration)
        self.writer.add_scalar("Image/Loss_L1", Loss_L1, iteration)
        self.writer.add_scalar("Image/Loss_VGG_tm", Loss_VGG_tm, iteration)
        self.writer.add_scalar("Image/Loss_small", Loss_small, iteration)
        self.writer.add_scalar("Image/DLoss", self.DLoss, iteration)
        
        whole_loss.backward()
        torch.nn.utils.clip_grad_norm(self.netG.parameters(), 10)
        self.optimizer_G.step()

        self.whole_loss = whole_loss.item()
        self.Loss_GAN = Loss_GAN.item()
        self.Loss_L1 = Loss_L1.item()
        self.Loss_VGG_tm = Loss_VGG_tm.item()
        self.Loss_small = Loss_small.item()

        self.m_LDR = self.img_sample['m_LDR']
        self.s_LDR = self.img_sample['s_LDR']
        self.l_LDR = self.img_sample['l_LDR']
        self.GT_HDR_tm = hdr_util.tonemap(self.img_sample['GT_HDR'])
        self.fake_HDR_tm = hdr_util.tonemap(fake_HDR.detach())

        self.img_psnr_u = compute_psnr(np.asarray(self.fake_HDR_tm[0].cpu()), np.asarray(self.GT_HDR_tm[0].cpu()))

        self.writer.add_scalar("Image/PSNR_u", self.img_psnr_u, iteration)
        

    def get_current_errors(self, iteration):
        ret_errors = OrderedDict([('whole_loss', self.whole_loss),
                                  ('Loss_GAN', self.Loss_GAN),
                                  ('Loss_L1', self.Loss_L1),
                                  ('Loss_VGG_tm', self.Loss_VGG_tm),
                                  ('Loss_small', self.Loss_small),
                                  ('DLoss', self.DLoss),
                                  ('PSNR_u', self.img_psnr_u)])
            
        return ret_errors

    
    def get_current_visuals_train(self, iteration):
        s_LDR = torchvision.utils.make_grid([self.s_LDR[[0]]])
        s_LDR = matplotlib_imshow(s_LDR[0])

        m_LDR = torchvision.utils.make_grid([self.m_LDR[[0]]])
        m_LDR = matplotlib_imshow(m_LDR[0])

        l_LDR = torchvision.utils.make_grid([self.l_LDR[[0]]])
        l_LDR = matplotlib_imshow(l_LDR[0])

        GT_HDR_tm = torchvision.utils.make_grid([self.GT_HDR_tm[[0]]])
        GT_HDR_tm = matplotlib_imshow(GT_HDR_tm[0])

        fake_HDR_tm = torchvision.utils.make_grid([self.fake_HDR_tm[[0]]])
        fake_HDR_tm = matplotlib_imshow(fake_HDR_tm[0])
        
        self.writer.add_image('Image/s_LDR', s_LDR)
        self.writer.add_image('Image/m_LDR', m_LDR)
        self.writer.add_image('Image/l_LDR', l_LDR)
        self.writer.add_image('Image/GT_HDR_tm', GT_HDR_tm)
        self.writer.add_image('Image/fake_HDR_tm', fake_HDR_tm)

        s_LDR = image_util.tensor2im(self.s_LDR)
        m_LDR = image_util.tensor2im(self.m_LDR)
        l_LDR = image_util.tensor2im(self.l_LDR)
        GT_HDR_tm = image_util.tensor2im(self.GT_HDR_tm)
        fake_HDR_tm = image_util.tensor2im(self.fake_HDR_tm)

        ret_visuals = OrderedDict([('s_LDR', s_LDR),
                                   ('m_LDR', m_LDR),
                                   ('l_LDR', l_LDR),
                                   ('GT_HDR_tm', GT_HDR_tm),
                                   ('fake_HDR_tm', fake_HDR_tm)])

        return ret_visuals


    def img_testHDR(self, img_test_loader, iteration):

        ### This part is only for test data or test data score computation and reserve
        f = open("HDRTest.txt","a+")
        f.write('iteration: %d'%iteration)

        ave_psnr_u = 0
        ave_psnr_L = 0

        img_num = 0

        self.netG.eval()

        with torch.no_grad():
            for i, data in enumerate(img_test_loader):
                fake_HDR, _ = self.netG([torch.cat([data['s_LDR'], data['s_HDR']], dim=1).cuda(),
                                         torch.cat([data['m_LDR'], data['m_HDR'],
                                                    data['bright_c'], data['dark_c']], dim=1).cuda(),
                                         torch.cat([data['l_LDR'], data['l_HDR']], dim=1).cuda()])
                # fake_HDR, _ = self.netG([torch.cat([data['s_LDR'], data['s_HDR']], dim=1),
                #                          torch.cat([data['m_LDR'], data['m_HDR'],
                #                                     data['bright_c'], data['dark_c']], dim=1),
                #                          torch.cat([data['l_LDR'], data['l_HDR']], dim=1)])
                fake_HDR_tm = hdr_util.tonemap(fake_HDR)
                GT_HDR_tm = hdr_util.tonemap(data['GT_HDR'])

                img_psnr_u = compute_psnr(np.asarray(fake_HDR_tm[0].cpu()), np.asarray(GT_HDR_tm[0]))
                img_psnr_L = compute_psnr(np.asarray(fake_HDR.cpu()), np.asarray(data['GT_HDR'][0]))

                
                ave_psnr_u += img_psnr_u
                ave_psnr_L += img_psnr_L

                fake_HDR_tm = image_util.tensor2im(fake_HDR_tm)

                cv2.imwrite('img_test_HDR/%d_fake_HDR_tm.png'%i, fake_HDR_tm)

                print('img_test_HDR/num:%04d image'%i)
                img_num += 1

        f.write('\n')
        f.write('Image_Num: {:d}, Average_psnr_u: {:4.4f}, Average_psnr_L: {:4.4f}'.format(
                img_num,
                ave_psnr_u/img_num,
                ave_psnr_L/img_num))  
        f.write('\n')
        f.write('\n')
        f.close()
        self.save_states_simple(self.netG, 'G', 'best', self.gpu_ids, self.device)
        self.netG.train()

        return
        

    def save(self, label):
        self.save_states(self.netG, self.optimizer_G, self.schedulers[0], 'G', label, self.gpu_ids, self.device)
        self.save_states(self.netD, self.optimizer_D, self.schedulers[1], 'D', label, self.gpu_ids, self.device)