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
from model.generator import Generator
import copy
from einops import repeat
from empatches import EMPatches
from pathlib import Path

# loss_fn_alex = lpips.LPIPS(net='alex').cuda()

def custom_l1_loss(fake, real):
    loss = torch.tensor(0.).cuda()
    loss += (torch.abs(fake-real)).sum()
    return loss

def custom_l2_loss(fake, real):
    ## Adopt the idea from RawNerf
    # loss = torch.tensor(0.).cuda()
    fake_for_gray = (fake.detach() + 1.) / 2
    fake_gray = (0.299 * fake_for_gray[:, 0] + 0.587 * fake_for_gray[:, 1] + 0.114 * fake_for_gray[:, 2])
    scaling_grad = 1. / (1e-3 + fake_gray)
    loss = (fake-real)**2 * scaling_grad**2
    return loss.mean()

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
        
class CustomNet(BaseModel):
    def name(self):
        return 'CustomNet'

    def initialize(self, opt):
        self.opt = opt
        BaseModel.initialize(self, opt)

        if self.isTrain:
            # For tensorboardX
            self.writer = SummaryWriter()
            # load/define networks
            self.netG = Generator(3, 5, 64, 32).to(self.device)

            # define loss functions
            self.criterionVGG = VGGLoss()
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

        fake_HDR, attn_entropy = self.netG([self.img_sample['s_HDR'],
                                            self.img_sample['m_HDR'],
                                            self.img_sample['l_HDR'],
                                            self.img_sample['s_LDR'],
                                            self.img_sample['m_LDR'],
                                            self.img_sample['l_LDR']])
        
        fake_HDR_tm = hdr_util.tonemap(fake_HDR)
        real_HDR_tm = hdr_util.tonemap(self.img_sample['GT_HDR'])
        
        Loss_L2_hdr_final = custom_l2_loss(fake_HDR, self.img_sample['GT_HDR'])
        Loss_VGG_tm = self.criterionVGG(fake_HDR_tm, real_HDR_tm) * 10
        Loss_attn = (attn_entropy - 0.05) ** 2

        whole_loss = Loss_L2_hdr_final + Loss_VGG_tm + Loss_attn

        self.writer.add_scalar("Image/Loss_L2_hdr_final", Loss_L2_hdr_final, iteration)
        self.writer.add_scalar("Image/Loss_VGG_tm", Loss_VGG_tm, iteration)
        self.writer.add_scalar("Image/Loss_attn", Loss_attn, iteration)

        whole_loss.backward()
        torch.nn.utils.clip_grad_norm(self.netG.parameters(), 1)
        self.optimizer_G.step()
        
        self.whole_loss = whole_loss.item()
        self.Loss_L2_hdr_final = Loss_L2_hdr_final.item()
        self.Loss_VGG_tm = Loss_VGG_tm.item()
        self.Loss_attn = Loss_attn.item()

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
            pseudo_GT, _ = ref_net(([self.img_sample['s_HDR_gt'],
                                     self.img_sample['m_HDR_gt'],
                                     self.img_sample['l_HDR_gt'],
                                     self.img_sample['s_LDR_gt'],
                                     self.img_sample['m_LDR_gt'],
                                     self.img_sample['l_LDR_gt']]))
        pseudo_GT = pseudo_GT.detach_()

        fake_HDR, attn_entropy = self.netG([self.img_sample['s_HDR'],
                                            self.img_sample['m_HDR'],
                                            self.img_sample['l_HDR'],
                                            self.img_sample['s_LDR'],
                                            self.img_sample['m_LDR'],
                                            self.img_sample['l_LDR']])

        fake_HDR_tm = hdr_util.tonemap(fake_HDR)
        pseudo_HDR_tm = hdr_util.tonemap(pseudo_GT)
        
        Loss_L2_hdr_final = custom_l2_loss(fake_HDR, pseudo_GT)
        Loss_VGG_tm = self.criterionVGG(fake_HDR_tm, pseudo_HDR_tm) * 10
        Loss_attn = (attn_entropy - 0.05) ** 2

        whole_loss = Loss_L2_hdr_final + Loss_VGG_tm + Loss_attn

        self.writer.add_scalar("Image/Loss_L2_hdr_final", Loss_L2_hdr_final, iteration)
        self.writer.add_scalar("Image/Loss_VGG_tm", Loss_VGG_tm, iteration)
        self.writer.add_scalar("Image/Loss_attn", Loss_attn, iteration)
        
        whole_loss.backward()
        torch.nn.utils.clip_grad_norm(self.netG.parameters(), 1)
        self.optimizer_G.step()

        self.whole_loss = whole_loss.item()
        self.Loss_L2_hdr_final = Loss_L2_hdr_final.item()
        self.Loss_VGG_tm = Loss_VGG_tm.item()
        self.Loss_attn = Loss_attn.item()

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
                                  ('Loss_L2_hdr_final', self.Loss_L2_hdr_final),
                                  ('Loss_VGG_tm', self.Loss_VGG_tm),
                                  ('Loss_attn', self.Loss_attn),
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


    def img_testHDR_full(self, img_test_loader, iteration, data_type):

        ### This part is only for test data or test data score computation and reserve

        img_num = 0
        if data_type == 'GT':
            f = open("HDRTest.txt","a+")
            f.write('iteration: %d'%iteration)
            
        dir_path = Path(f'img_test_HDR_{data_type}')

        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

        CustomNet = Generator(3, 5, 64, 32)

        self.load_states_simple(CustomNet, 'G', iteration)
        
        CustomNet.eval()
        CustomNet = CustomNet

        networks.print_network(CustomNet)

        with torch.no_grad():
            for i, data in enumerate(img_test_loader):
                img_psnr_L = 0
                img_psnr_u = 0
                img_ssim_L = 0
                img_ssim_u = 0

                torch.cuda.synchronize()
                start = time.time()
                
                fake_HDR, _ = CustomNet([data['s_HDR'],
                                         data['m_HDR'],
                                         data['l_HDR'],
                                         data['s_LDR'],
                                         data['m_LDR'],
                                         data['l_LDR']],
                                    )

                fake_HDR_tp = hdr_util.tonemap(fake_HDR.detach())
                if data_type == 'GT':
                    real_HDR_tp = hdr_util.tonemap(data['GT_HDR'])
                
                if data_type == 'GT':
                    img_psnr_L += compute_psnr(np.asarray(fake_HDR.cpu()), np.asarray(data['GT_HDR']))
                    img_psnr_u += compute_psnr(np.asarray(fake_HDR_tp.cpu()), np.asarray(real_HDR_tp))

                    img_ssim_L += compute_ssim(np.asarray(fake_HDR[0].cpu()).transpose(1,2,0), np.asarray(data['GT_HDR'][0]).transpose(1,2,0))
                    img_ssim_u += compute_ssim(np.asarray(fake_HDR_tp[0].cpu()).transpose(1,2,0), np.asarray(real_HDR_tp[0]).transpose(1,2,0))
                    
                    f.write('\n')
                    f.write('Image_No: {:d}, PSNR_L: {:4.4f}, PSNR_u: {:4.4f}, SSIM_L: {:4.4f}, SSIM_u: {:4.4f}'.format(i, img_psnr_L, img_psnr_u, img_ssim_L, img_ssim_u))
                    f.write('\n')

                fake_HDR = image_util.tensor2im(fake_HDR, 'HDR', np.float32)
                if data_type == 'GT':
                    real_HDR = image_util.tensor2im(data['GT_HDR'], 'HDR', np.float32)

                cv2.imwrite('img_test_HDR_%s/%d_fake_HDR.hdr'%(data_type, i), fake_HDR)
                if data_type == 'GT':
                    cv2.imwrite('img_test_HDR_%s/%d_real_HDR.hdr'%(data_type, i), real_HDR)
                
                print('img_test_HDR/num:%04d image'%i)
                img_num += 1

        if data_type == 'GT':
            f.close()
            
        return
    

    def img_testHDR_full_linear_merge(self, img_test_loader, iteration, data_type):

        ### This function is designed for processing super high-resolution input

        ## network init
        CustomNet = Generator(3, 5, 64, 32)
        self.load_states_simple(CustomNet, 'G', iteration)
        CustomNet.eval()
        CustomNet = CustomNet.cuda()

        ## result path init
        if data_type == 'GT':
            f = open("HDRTest.txt","a+")
            f.write('iteration: %d'%iteration)
        dir_path = Path(f'img_test_HDR_{data_type}')
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

        img_num = 0
        emp = EMPatches()
        with torch.no_grad():
            for i, data in enumerate(img_test_loader):
                img_psnr_L = 0
                img_psnr_u = 0
                img_ssim_L = 0
                img_ssim_u = 0

                torch.cuda.synchronize()
                start = time.time()

                small_s_hdr_patches, indices = emp.extract_patches(np.asarray(data['s_HDR'][0]).transpose(1,2,0), patchsize=800, stride=600)
                small_m_hdr_patches, indices = emp.extract_patches(np.asarray(data['m_HDR'][0]).transpose(1,2,0), patchsize=800, stride=600)
                small_l_hdr_patches, indices = emp.extract_patches(np.asarray(data['l_HDR'][0]).transpose(1,2,0), patchsize=800, stride=600)
                
                small_s_ldr_patches, indices = emp.extract_patches(np.asarray(data['s_LDR'][0]).transpose(1,2,0), patchsize=800, stride=600)
                small_m_ldr_patches, indices = emp.extract_patches(np.asarray(data['m_LDR'][0]).transpose(1,2,0), patchsize=800, stride=600)
                small_l_ldr_patches, indices = emp.extract_patches(np.asarray(data['l_LDR'][0]).transpose(1,2,0), patchsize=800, stride=600)

                small_fake_HDRs = []
                
                for small_s_hdr, small_m_hdr, small_l_hdr, small_s_ldr, small_m_ldr, small_l_ldr in\
                    zip(small_s_hdr_patches, small_m_hdr_patches, small_l_hdr_patches, small_s_ldr_patches, small_m_ldr_patches, small_l_ldr_patches):
                
                    small_fake_HDR, _ = CustomNet([torch.from_numpy(small_s_hdr.transpose(2,0,1)).unsqueeze(0).cuda(),
                                                   torch.from_numpy(small_m_hdr.transpose(2,0,1)).unsqueeze(0).cuda(),
                                                   torch.from_numpy(small_l_hdr.transpose(2,0,1)).unsqueeze(0).cuda(),
                                                   torch.from_numpy(small_s_ldr.transpose(2,0,1)).unsqueeze(0).cuda(),
                                                   torch.from_numpy(small_m_ldr.transpose(2,0,1)).unsqueeze(0).cuda(),
                                                   torch.from_numpy(small_l_ldr.transpose(2,0,1)).unsqueeze(0).cuda()])

                    small_fake_HDRs.append(small_fake_HDR[0].cpu().numpy().transpose(1,2,0))

                fake_HDR = emp.merge_patches(small_fake_HDRs, indices, mode='avg')
                fake_HDR = torch.from_numpy(fake_HDR.transpose(2,0,1)).unsqueeze(0)
                fake_HDR_tp = hdr_util.tonemap(fake_HDR.detach())
                if data_type == 'GT':
                    real_HDR_tp = hdr_util.tonemap(data['GT_HDR'])
                
                if data_type == 'GT':
                    img_psnr_L += compute_psnr(np.asarray(fake_HDR.cpu()), np.asarray(data['GT_HDR']))
                    img_psnr_u += compute_psnr(np.asarray(fake_HDR_tp.cpu()), np.asarray(real_HDR_tp))

                    img_ssim_L += compute_ssim(np.asarray(fake_HDR[0].cpu()).transpose(1,2,0), np.asarray(data['GT_HDR'][0]).transpose(1,2,0))
                    img_ssim_u += compute_ssim(np.asarray(fake_HDR_tp[0].cpu()).transpose(1,2,0), np.asarray(real_HDR_tp[0]).transpose(1,2,0))
                    
                    f.write('\n')
                    f.write('Image_No: {:d}, PSNR_L: {:4.4f}, PSNR_u: {:4.4f}, SSIM_L: {:4.4f}, SSIM_u: {:4.4f}'.format(i, img_psnr_L, img_psnr_u, img_ssim_L, img_ssim_u))
                    f.write('\n')

                fake_HDR = image_util.tensor2im(fake_HDR, 'HDR', np.float32)
                if data_type == 'GT':
                    real_HDR = image_util.tensor2im(data['GT_HDR'], 'HDR', np.float32)

                cv2.imwrite('img_test_HDR_%s/%d_fake_HDR.hdr'%(data_type, i), fake_HDR)
                if data_type == 'GT':
                    cv2.imwrite('img_test_HDR_%s/%d_real_HDR.hdr'%(data_type, i), real_HDR)
                
                print('img_test_HDR/num:%04d image'%i)
                img_num += 1

        if data_type == 'GT':
            f.close()
            
        return

    def test_speed(self, iteration):
        CustomNet = Generator(3, 5, 64, 32)

        CustomNet.eval()
        CustomNet = CustomNet.cuda()

        input_hdr = torch.rand((1, 3, 2048, 1080))

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            otuput_hdr, _ = CustomNet([input_hdr.cuda(), input_hdr.cuda(), input_hdr.cuda(),
                                       input_hdr.cuda(), input_hdr.cuda(), input_hdr.cuda()])
        
        torch.cuda.synchronize()
        end = time.time()
        print(end-start)
        pytorch_total_params = sum(p.numel() for p in CustomNet.parameters())
        print('Network param number: %d'%pytorch_total_params)
        print(otuput_hdr.shape)
        

    def save(self, label):
        self.save_states(self.netG, self.optimizer_G, self.schedulers[0], 'G', label, self.gpu_ids, self.device)