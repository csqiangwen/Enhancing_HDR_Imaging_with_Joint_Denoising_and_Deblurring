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
from model.eval_tools import compute_psnr, compute_ssim
import torch.nn.functional as F
import cv2
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
import lpips
import os
import random
import time
from model.attnnet import AttnNet
import copy

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

def my_sqrt(ten):
    ten = (ten + 1.) / 2.
    ten  = torch.sqrt(ten)
    ten = (ten - 0.5) * 2.
    return ten

def my_desqrt(ten):
    ten = (ten + 1.) / 2.
    ten  = ten ** 2
    ten = (ten - 0.5) * 2.
    return ten
        
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
            self.netG = AttnNet(3, 5, 64, 32).to(self.device)

            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionTV = model_util.TVLoss()
            self.criterionTM = model_util.mu_loss()
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
                
            self.netG.train()

            networks.print_network(self.netG)
        
        if not self.isTrain:
            pass


    def img_train_forward(self, data):
        self.img_sample = {}
        self.img_sample['GT_HDR'] = data['GT_HDR'].to(self.device)
    
        self.img_sample['s_LDR'] = data['s_LDR'].to(self.device)
        self.img_sample['m_LDR'] = data['m_LDR'].to(self.device)
        self.img_sample['l_LDR'] = data['l_LDR'].to(self.device)

        self.img_sample['s_HDR'] = data['s_HDR'].to(self.device)
        self.img_sample['m_HDR'] = data['m_HDR'].to(self.device)
        self.img_sample['l_HDR'] = data['l_HDR'].to(self.device)

        self.img_sample['s_LDR_gt'] = data['s_LDR_gt'].to(self.device)
        self.img_sample['m_LDR_gt'] = data['m_LDR_gt'].to(self.device)
        self.img_sample['l_LDR_gt'] = data['l_LDR_gt'].to(self.device)

        self.img_sample['s_HDR_gt'] = data['s_HDR_gt'].to(self.device)
        self.img_sample['m_HDR_gt'] = data['m_HDR_gt'].to(self.device)
        self.img_sample['l_HDR_gt'] = data['l_HDR_gt'].to(self.device)


    def random_crop(self, input, size):
        _, _, h, w = input.shape
        x, y = random.randint(0, h-size), random.randint(0, w-size)
        return input[:, :, x:x+size, y:y+size], x, y

    def crop(self, input, coords, size):
        return input[:, :, coords[0]:coords[0]+size, coords[1]:coords[1]+size]


    def img_optimize_parameters_static(self, iteration):

        self.optimizer_G.zero_grad()
        mid_fake_HDR, fake_HDR = self.netG([self.img_sample['s_HDR'],
                                            self.img_sample['m_HDR'],
                                            self.img_sample['l_HDR'],
                                            self.img_sample['s_LDR'],
                                            self.img_sample['m_LDR'],
                                            self.img_sample['l_LDR']])

        fake_HDR_tm = hdr_util.tonemap(fake_HDR)
        real_HDR_tm = hdr_util.tonemap(self.img_sample['GT_HDR'])
        
        Loss_L2_hdr_mid = self.criterionL2(mid_fake_HDR, self.img_sample['GT_HDR']) * 10
        Loss_L1_hdr_final = self.criterionL1(fake_HDR, self.img_sample['GT_HDR'])
        Loss_L1_tm = self.criterionL1(fake_HDR_tm, real_HDR_tm)

        whole_loss = Loss_L1_hdr_final + Loss_L1_tm + Loss_L2_hdr_mid

        self.writer.add_scalar("Image/Loss_L1_hdr_final", Loss_L1_hdr_final, iteration)
        self.writer.add_scalar("Image/Loss_L1_tm", Loss_L1_tm, iteration)
        self.writer.add_scalar("Image/Loss_L2_hdr_mid", Loss_L2_hdr_mid, iteration)

        whole_loss.backward()
        self.optimizer_G.step()
        
        self.whole_loss = whole_loss.item()
        self.Loss_L1_hdr_final = Loss_L1_hdr_final.item()
        self.Loss_L1_tm = Loss_L1_tm.item()
        self.Loss_L2_hdr_mid = Loss_L2_hdr_mid.item()

        self.s_LDR = self.img_sample['s_LDR']
        self.m_LDR = self.img_sample['m_LDR']
        self.l_LDR = self.img_sample['l_LDR']
        self.GT_HDR_tm = real_HDR_tm
        self.fake_HDR_tm = fake_HDR_tm.detach()

        self.PSNR_L = compute_psnr(np.asarray(fake_HDR.detach().cpu()), np.asarray(self.img_sample['GT_HDR'].cpu()))
        self.PSNR_u = compute_psnr(np.asarray(fake_HDR_tm.detach().cpu()), np.asarray(real_HDR_tm.cpu()))

        self.writer.add_scalar("Image/PSNR_L", self.PSNR_L, iteration)
        self.writer.add_scalar("Image/PSNR_u", self.PSNR_u, iteration)


    def img_optimize_parameters_dynamic(self, iteration, ref_net):

        self.optimizer_G.zero_grad()

        with torch.no_grad():
            _, pseudo_GT = ref_net(([self.img_sample['s_HDR_gt'],
                                     self.img_sample['m_HDR_gt'],
                                     self.img_sample['l_HDR_gt'],
                                     self.img_sample['s_LDR_gt'],
                                     self.img_sample['m_LDR_gt'],
                                     self.img_sample['l_LDR_gt']]))
        pseudo_GT = pseudo_GT.detach_()

        _, fake_HDR = self.netG([self.img_sample['s_HDR'],
                                            self.img_sample['m_HDR'],
                                            self.img_sample['l_HDR'],
                                            self.img_sample['s_LDR'],
                                            self.img_sample['m_LDR'],
                                            self.img_sample['l_LDR']])

        fake_HDR_tm = hdr_util.tonemap(fake_HDR)
        pseudo_HDR_tm = hdr_util.tonemap(pseudo_GT)
        
        # Loss_L2_hdr_mid = self.criterionL2(mid_fake_HDR, self.img_sample['GT_HDR']) * 10
        Loss_L1_hdr_final = self.criterionL1(fake_HDR, pseudo_GT)
        Loss_L1_tm = self.criterionL1(fake_HDR_tm, pseudo_HDR_tm)

        whole_loss = Loss_L1_hdr_final + Loss_L1_tm

        self.writer.add_scalar("Image/Loss_L1_hdr_final", Loss_L1_hdr_final, iteration)
        self.writer.add_scalar("Image/Loss_L1_tm", Loss_L1_tm, iteration)
        
        whole_loss.backward()
        self.optimizer_G.step()

        self.whole_loss = whole_loss.item()
        self.Loss_L1_hdr_final = Loss_L1_hdr_final.item()
        self.Loss_L1_tm = Loss_L1_tm.item()

        self.m_LDR = self.img_sample['m_LDR']
        self.s_LDR = self.img_sample['s_LDR']
        self.l_LDR = self.img_sample['l_LDR']
        self.GT_HDR_tm = pseudo_HDR_tm
        self.fake_HDR_tm = hdr_util.tonemap(fake_HDR.detach())

        self.PSNR_L = compute_psnr(np.asarray(fake_HDR.detach().cpu()), np.asarray(pseudo_GT.cpu()))
        self.PSNR_u = compute_psnr(np.asarray(fake_HDR_tm.detach().cpu()), np.asarray(pseudo_HDR_tm.cpu()))

        self.writer.add_scalar("Image/PSNR_L", self.PSNR_L, iteration)
        self.writer.add_scalar("Image/PSNR_u", self.PSNR_u, iteration)
        

    def get_current_errors(self, iteration):
        ret_errors = OrderedDict([('whole_loss', self.whole_loss),
                                  ('Loss_L1_hdr_final', self.Loss_L1_hdr_final),
                                  ('Loss_L1_tm', self.Loss_L1_tm),
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
        f = open("HDRTest.txt","a+")
        f.write('iteration: %d'%iteration)

        netHDR = AttnNet(3, 5, 64, 32)

        self.load_states_simple(netHDR, 'G', iteration)
        
        netHDR.eval()
        netHDR = netHDR.cuda()

        with torch.no_grad():
            for i, data in enumerate(img_test_loader):
                torch.cuda.synchronize()
                start = time.time()
                LDR_shape = data['s_LDR'].shape
                canvas_hdr = torch.ones([1, 3, LDR_shape[2], LDR_shape[3]]).cuda()
                canvas_tp = torch.ones([1, 3, LDR_shape[2], LDR_shape[3]]).cuda()
                real_HDR_tp = hdr_util.tonemap(data['GT_HDR'])
                img_psnr_L = 0
                img_psnr_u = 0
                img_ssim_L = 0
                img_ssim_u = 0
                for m in range(4):
                    for n in range(4):
                        small_s_ldr = data['s_LDR'][:, :, LDR_shape[2]//4*m:LDR_shape[2]//4*(m+1), LDR_shape[3]//4*n:LDR_shape[3]//4*(n+1)].cuda()
                        small_m_ldr = data['m_LDR'][:, :, LDR_shape[2]//4*m:LDR_shape[2]//4*(m+1), LDR_shape[3]//4*n:LDR_shape[3]//4*(n+1)].cuda()
                        small_l_ldr = data['l_LDR'][:, :, LDR_shape[2]//4*m:LDR_shape[2]//4*(m+1), LDR_shape[3]//4*n:LDR_shape[3]//4*(n+1)].cuda()

                        small_s_hdr = data['s_HDR'][:, :, LDR_shape[2]//4*m:LDR_shape[2]//4*(m+1), LDR_shape[3]//4*n:LDR_shape[3]//4*(n+1)].cuda()
                        small_m_hdr = data['m_HDR'][:, :, LDR_shape[2]//4*m:LDR_shape[2]//4*(m+1), LDR_shape[3]//4*n:LDR_shape[3]//4*(n+1)].cuda()
                        small_l_hdr = data['l_HDR'][:, :, LDR_shape[2]//4*m:LDR_shape[2]//4*(m+1), LDR_shape[3]//4*n:LDR_shape[3]//4*(n+1)].cuda()

                        _, small_fake_HDR = netHDR([small_s_hdr,
                                                    small_m_hdr,
                                                    small_l_hdr,
                                                    small_s_ldr,
                                                    small_m_ldr,
                                                    small_l_ldr])

                        fake_HDR_tp = hdr_util.tonemap(small_fake_HDR.detach())

                        canvas_hdr[:, :, LDR_shape[2]//4*m:LDR_shape[2]//4*(m+1), LDR_shape[3]//4*n:LDR_shape[3]//4*(n+1)] = small_fake_HDR
                        canvas_tp[:, :, LDR_shape[2]//4*m:LDR_shape[2]//4*(m+1), LDR_shape[3]//4*n:LDR_shape[3]//4*(n+1)] = fake_HDR_tp
                
                img_psnr_L += compute_psnr(np.asarray(canvas_hdr.cpu()), np.asarray(data['GT_HDR']))
                img_psnr_u += compute_psnr(np.asarray(canvas_tp.cpu()), np.asarray(real_HDR_tp))

                img_ssim_L += compute_ssim(np.asarray(canvas_hdr[0].cpu()).transpose(1,2,0), np.asarray(data['GT_HDR'][0]).transpose(1,2,0))
                img_ssim_u += compute_ssim(np.asarray(canvas_tp[0].cpu()).transpose(1,2,0), np.asarray(real_HDR_tp[0]).transpose(1,2,0))
                    
                f.write('\n')
                f.write('Image_No: {:d}, PSNR_L: {:4.4f}, PSNR_u: {:4.4f}, SSIM_L: {:4.4f}, SSIM_u: {:4.4f}'.format(i, img_psnr_L, img_psnr_u, img_ssim_L, img_ssim_u))
                f.write('\n')

                torch.cuda.synchronize()
                end = time.time()
                print(end-start)
                print(data['s_LDR'].shape)
                
                fake_HDR = image_util.tensor2im(canvas_hdr, 'HDR', np.float32)
                real_HDR = image_util.tensor2im(data['GT_HDR'], 'HDR', np.float32)

                cv2.imwrite('img_test_HDR_%s/%d_fake_HDR.hdr'%(data_type, i), fake_HDR)
                cv2.imwrite('img_test_HDR_%s/%d_real_HDR.hdr'%(data_type, i), real_HDR)

                print('img_test_HDR/num:%04d image'%i)
                img_num += 1

        f.close()

        return

    def img_testHDR_full(self, img_test_loader, iteration, data_type):

        ### This part is only for test data or test data score computation and reserve

        img_num = 0
        f = open("HDRTest.txt","a+")
        f.write('iteration: %d'%iteration)

        netHDR = AttnNet(3, 5, 64, 32)

        self.load_states_simple(netHDR, 'G', iteration)
        
        netHDR.eval()
        netHDR = netHDR.cuda()

        with torch.no_grad():
            for i, data in enumerate(img_test_loader):
                torch.cuda.synchronize()
                start = time.time()

                _, fake_HDR = netHDR([data['s_LDR'].cuda(),
                                      data['m_LDR'].cuda(),
                                      data['l_LDR'].cuda()])

                print(fake_HDR.shape)

                torch.cuda.synchronize()
                end = time.time()
                print(end-start)
                print(data['s_LDR'].shape)
                
                print('img_test_HDR/num:%04d image'%i)
                img_num += 1

        return

    def img_testHDR_sensetime(self, img_test_loader, iteration):

        ### This part is only for test data or test data score computation and reserve

        img_num = 0
        netG = AttnNet(3, 5, 64, 32)

        self.load_states_simple(netG, 'G', iteration)
        
        netG.eval()
        netG = netG.cuda()

        with torch.no_grad():
            for i, data in enumerate(img_test_loader):
                torch.cuda.synchronize()
                start = time.time()
                LDR_shape = data['m_LDR'].shape
                canvas = torch.ones([1, 3, LDR_shape[2], LDR_shape[3]]).cuda()
                for m in range(4):
                    for n in range(4):
                        small_s_ldr = data['s_LDR'][:, :, LDR_shape[2]//4*m:LDR_shape[2]//4*(m+1), LDR_shape[3]//4*n:LDR_shape[3]//4*(n+1)]
                        small_m_ldr = data['m_LDR'][:, :, LDR_shape[2]//4*m:LDR_shape[2]//4*(m+1), LDR_shape[3]//4*n:LDR_shape[3]//4*(n+1)]
                        small_l_ldr = data['l_LDR'][:, :, LDR_shape[2]//4*m:LDR_shape[2]//4*(m+1), LDR_shape[3]//4*n:LDR_shape[3]//4*(n+1)]

                        _, small_fake_HDR = netG([small_s_ldr.cuda(),
                                                  small_m_ldr.cuda(),
                                                  small_l_ldr.cuda()])
                        canvas[:, :, LDR_shape[2]//4*m:LDR_shape[2]//4*(m+1), LDR_shape[3]//4*n:LDR_shape[3]//4*(n+1)] = small_fake_HDR
                torch.cuda.synchronize()
                end = time.time()
                print(end-start)
                
                fake_HDR = image_util.tensor2im(canvas, 'HDR', np.float32)

                cv2.imwrite('img_test_HDR_%s/%d_fake_HDR.hdr'%('sensetime', i), fake_HDR)

                print('img_test_HDR/num:%04d image'%i)
                img_num += 1

        return
        

    def save(self, label):
        self.save_states(self.netG, self.optimizer_G, self.schedulers[0], 'G', label, self.gpu_ids, self.device)