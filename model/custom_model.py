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
import random
import time

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
            self.netG = networks.define_G(in_channel=3).to(self.device)

            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionGAN = networks.GANLoss(gan_mode='lsgan').to(self.device)
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionTV = model_util.TVLoss()
            self.criterionVGG = VGGLoss()
            self.imgGRAD = model_util.ImageGradient().cuda()
            self.imgSmoothing = model_util.ImageSmoothing().cuda()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.networks = []
            self.optimizers = []
            self.schedulers = []
            self.networks.append(self.netG)
            self.optimizers.append(self.optimizer_G)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))
                
            if opt.continue_train:
                which_iter = opt.which_iter
                self.load_states(self.netG, self.optimizer_G, self.schedulers[0], 'G', which_iter)
                
            if len(self.gpu_ids) > 0:
                # self.netG = DataParallel(self.netG)
                pass
                
            self.netG.train()
        
        if not self.isTrain:
            # load/define networks
            self.netG = networks.define_G(in_channel=3)

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

    def random_crop(self, input, size):
        _, _, h, w = input.shape
        x, y = random.randint(0, h-size), random.randint(0, w-size)
        return input[:, :, x:x+size, y:y+size], x, y

    def crop(self, input, coords, size):
        return input[:, :, coords[0]:coords[0]+size, coords[1]:coords[1]+size]


    def img_optimize_parameters(self, iteration, data_type):

        ### input shape [B, C, H, W]
        B, C, H, W = self.img_sample['s_HDR'].shape

        self.optimizer_G.zero_grad()

        fake_HDR = self.netG([self.img_sample['s_LDR'],
                              self.img_sample['m_LDR'],
                              self.img_sample['l_LDR']])
        fake_HDR_tp = hdr_util.tonemap(fake_HDR)
        real_HDR_tp = hdr_util.tonemap(self.img_sample['GT_HDR'])
        

        if data_type == 'static':
            Loss_VGG = self.criterionVGG(fake_HDR, self.img_sample['GT_HDR']) + \
                       self.criterionVGG(fake_HDR_tp, real_HDR_tp)
            Loss_L1 = self.criterionL1(fake_HDR, self.img_sample['GT_HDR']) + \
                      self.criterionL1(fake_HDR_tp, real_HDR_tp)
        else:
            Loss_VGG = self.criterionVGG(self.imgSmoothing(fake_HDR), self.imgSmoothing(self.img_sample['GT_HDR'])) +\
                       self.criterionVGG(self.imgSmoothing(fake_HDR_tp), self.imgSmoothing(real_HDR_tp))
            Loss_L1 = Loss_VGG * 0        
        
        whole_loss = Loss_VGG + Loss_L1

        self.writer.add_scalar("Image/Loss_VGG", Loss_VGG, iteration)
        self.writer.add_scalar("Image/Loss_L1", Loss_L1, iteration)
        
        whole_loss.backward()
        torch.nn.utils.clip_grad_norm(self.netG.parameters(), 10)
        self.optimizer_G.step()

        self.whole_loss = whole_loss.item()
        self.Loss_VGG = Loss_VGG.item()
        self.Loss_L1 = Loss_L1.item()

        self.m_LDR = self.img_sample['m_LDR']
        self.s_LDR = self.img_sample['s_LDR']
        self.l_LDR = self.img_sample['l_LDR']
        self.GT_HDR_tm = real_HDR_tp
        self.fake_HDR_tm = hdr_util.tonemap(fake_HDR.detach())

        self.PSNR_L = compute_psnr(np.asarray(fake_HDR[0].detach().cpu()), np.asarray(self.img_sample['GT_HDR'].cpu()))
        self.PSNR_u = compute_psnr(np.asarray(fake_HDR_tp.detach().cpu()), np.asarray(real_HDR_tp.cpu()))

        self.writer.add_scalar("Image/PSNR_L", self.PSNR_L, iteration)
        self.writer.add_scalar("Image/PSNR_u", self.PSNR_u, iteration)
        

    def get_current_errors(self, iteration):
        ret_errors = OrderedDict([('whole_loss', self.whole_loss),
                                  ('Loss_VGG', self.Loss_VGG),
                                  ('Loss_L1', self.Loss_L1),
                                  ('PSNR_L', self.PSNR_L),
                                  ('PSNR_u', self.PSNR_u)])
            
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


    def img_testHDR(self, img_test_loader, iteration, data_type):

        ### This part is only for test data or test data score computation and reserve

        img_num = 0
        if data_type == 'static':
            f = open("HDRTest.txt","a+")
            f.write('iteration: %d'%iteration)

        self.netG.eval()

        with torch.no_grad():
            for i, data in enumerate(img_test_loader):
                torch.cuda.synchronize()
                start = time.time()
                fake_HDR = self.netG([data['s_LDR'].cuda(),
                                      data['m_LDR'].cuda(),
                                      data['l_LDR'].cuda()])

                fake_HDR_tp = hdr_util.tonemap(fake_HDR)
                real_HDR_tp = hdr_util.tonemap(data['GT_HDR'][0])

                torch.cuda.synchronize()
                end = time.time()
                print(end-start)
                print(data['s_LDR'].shape)

                if data_type == 'static':
                    img_psnr_L = compute_psnr(np.asarray(fake_HDR.cpu()), np.asarray(data['GT_HDR'][0]))
                    img_psnr_u = compute_psnr(np.asarray(fake_HDR_tp.cpu()), np.asarray(real_HDR_tp))

                    f.write('\n')
                    f.write('Image_No: {:d}, PSNR_L: {:4.4f}, PSNR_u: {:4.4f}'.format(i, img_psnr_L, img_psnr_u))
                    f.write('\n')
                
                fake_HDR = image_util.tensor2im(fake_HDR, 'HDR', np.float32)
                real_HDR = image_util.tensor2im(data['GT_HDR'], 'HDR', np.float32)

                cv2.imwrite('img_test_HDR_%s/%d_fake_HDR.hdr'%(data_type, i), fake_HDR)
                cv2.imwrite('img_test_HDR_%s/%d_real_HDR.hdr'%(data_type, i), real_HDR)

                print('img_test_HDR/num:%04d image'%i)
                img_num += 1

        self.save_states_simple(self.netG, 'G', 'best', self.gpu_ids, self.device)
        self.netG.train()

        if data_type == 'static':
            f.close()

        return

    def img_testHDR_sensetime(self, img_test_loader, iteration):

        ### This part is only for test data or test data score computation and reserve

        img_num = 0
        self.netG.eval()

        with torch.no_grad():
            for i, data in enumerate(img_test_loader):
                torch.cuda.synchronize()
                start = time.time()
                coarse_fake_HDR, refine_fake_HDR = self.netG([data['s_LDR'],
                                                              data['m_LDR'],
                                                              data['l_LDR']])
                torch.cuda.synchronize()
                end = time.time()
                print(end-start)
                
                coarse_fake_HDR = image_util.tensor2im(coarse_fake_HDR, 'HDR', np.float32)
                refine_fake_HDR = image_util.tensor2im(refine_fake_HDR, 'HDR', np.float32)

                cv2.imwrite('img_test_HDR_%s/%d_coarse_fake_HDR.hdr'%('sensetime', i), coarse_fake_HDR)
                cv2.imwrite('img_test_HDR_%s/%d_refine_fake_HDR.hdr'%('sensetime', i), refine_fake_HDR)

                print('img_test_HDR/num:%04d image'%i)
                img_num += 1

        return
        

    def save(self, label):
        self.save_states(self.netG, self.optimizer_G, self.schedulers[0], 'G', label, self.gpu_ids, self.device)