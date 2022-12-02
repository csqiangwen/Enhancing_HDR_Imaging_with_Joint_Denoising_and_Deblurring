import os.path
import torch
import torch.utils.data as data
import random
import torchvision.transforms as transforms
import data.util as util
import util.hdr_util as hdr_util
import cv2
from natsort import natsorted
import glob
import numpy as np
from pathlib import Path

def _get_params(param_file):
    with open(param_file) as fr:
        exps = fr.read().split('\n')[:3]
        iso = fr.read().split('\n')[3]
    # return [2. ** float(i) for i in exps]
    return [float(i) for i in exps], iso

def ev_alignment(img, expo, gamma):
    return ((img ** gamma) * 2.0**(-1*expo))**(1/gamma)

## Dataloader for RainDrop dataset
class HDRDataset(data.Dataset):
    def __init__(self, opt):
        super(HDRDataset,self).__init__()
        self.opt = opt
        root_path = Path(self.opt.hdr_dararoot)
        self.folder_path_1 = root_path / 'train_sharp' / 'static_new'
        self.folder_list_1 = natsorted(os.listdir(self.folder_path_1))

        self.param_path = root_path / 'train_sharp' / 'static_info'

        self.img_num = len(self.folder_list_1)

    def random_crop_flip(self, img_list, opt):
        if random.random()>0.2:
            h,w = img_list[0].shape[0], img_list[0].shape[1]
            h_ind = random.randint(0, h-self.opt.loadsize-1)
            w_ind = random.randint(0, w-self.opt.loadsize-1)
            
            for i, img in enumerate(img_list):
                img_list[i] = img[h_ind:h_ind+opt.loadsize, w_ind:w_ind+opt.loadsize, :]
        else:
            h,w = img_list[0].shape[0], img_list[0].shape[1]
            if h <= w:
                crop_size = h
                h_ind = 0
                w_ind = random.randint(0, w-crop_size-1)
                for i, img in enumerate(img_list):
                    img_list[i] = cv2.resize(img[:, w_ind:w_ind+crop_size, :],
                                            (opt.loadsize, opt.loadsize), interpolation=cv2.INTER_LINEAR)
            else:
                crop_size = w
                h_ind = random.randint(0, h-crop_size-1)
                w_ind = 0
                for i, img in enumerate(img_list):
                    img_list[i] = cv2.resize(img[h_ind:h_ind+crop_size, :, :],
                                            (opt.loadsize, opt.loadsize), interpolation=cv2.INTER_LINEAR)

        if random.random() > 0.5:
            for i, img in enumerate(img_list):
                # horizontal
                img_list[i] = cv2.flip(img, 1)
            
        if random.random() > 0.5:
            for i, img in enumerate(img_list):
                # vertical
                img_list[i] = cv2.flip(img, 0)
        
        return img_list


    def gaussian_blur(self, img_list):
        new_list = []
        for i, img in enumerate(img_list):
            kernel_size = random.sample([5, 7, 9, 11], 1)[0]
            new_list.append(cv2.GaussianBlur(img,(kernel_size, kernel_size),0))
        return new_list


    def blur_gen(self, img, motion_type):
        kernel_size = random.sample([20, 25, 30], 1)[0]

        # Create the vertical kernel.
        kernel_v = np.zeros((kernel_size, kernel_size))

        # Create a copy of the same for creating the horizontal kernel.
        kernel_h = np.copy(kernel_v)

        # Fill the middle row with ones.
        kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
        kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
        
        # Normalize.
        kernel_v /= kernel_size
        kernel_h /= kernel_size

        if motion_type == 'v':
            return cv2.filter2D(img, -1, kernel_v)
        else:
            return cv2.filter2D(img, -1, kernel_h)


    def motion_blur(self, img_list):
        new_list = []
        motion_type = None
        if random.random() > 0.5:
            motion_type = 'v'
        else:
            motion_type = 'h'
        for i, img in enumerate(img_list):
            new_list.append(self.blur_gen(img, motion_type))
        return new_list


    def __getitem__(self, index):

        img_path = self.folder_path_1 / self.folder_list_1[index]

        # expos, iso = _get_params(self.param_path / (self.folder_list_1[index]+'.txt'))
        
        # Read images
        [s_LDR, m_LDR, l_LDR, s_LDR_gt, m_LDR_gt, l_LDR_gt, GT_HDR] = [cv2.imread(str(img_path / 'input' / 'short.tif')),
                                                                       cv2.imread(str(img_path / 'input' / 'medium.tif')),
                                                                       cv2.imread(str(img_path / 'input' / 'long.tif')),
                                                                       cv2.imread(str(img_path / 'gt' / 'short.tif')),
                                                                       cv2.imread(str(img_path / 'gt' / 'medium.tif')),
                                                                       cv2.imread(str(img_path / 'gt' / 'long.tif')),
                                                                       cv2.imread(str(img_path / 'gt' / 'HDR_p_norm.hdr'), flags=cv2.IMREAD_ANYDEPTH)]
        [s_LDR, m_LDR, l_LDR, s_LDR_gt, m_LDR_gt, l_LDR_gt, GT_HDR] = self.random_crop_flip([s_LDR, m_LDR, l_LDR, s_LDR_gt, m_LDR_gt, l_LDR_gt, GT_HDR], self.opt)
        if random.random() > 0.5:
            [m_noise, l_noise] = [m_LDR-m_LDR_gt, l_LDR-l_LDR_gt]
            [m_LDR, l_LDR] = self.motion_blur([m_LDR_gt, l_LDR_gt])
            [m_LDR, l_LDR] = [np.clip(m_LDR+m_noise, 0, 255), np.clip(l_LDR+l_noise, 0, 255)]
        else:
            pass
        [s_LDR, m_LDR, l_LDR, s_LDR_gt, m_LDR_gt, l_LDR_gt, GT_HDR] = [(cv2.cvtColor(s_LDR, cv2.COLOR_BGR2RGB) / 255.),
                                                                       (cv2.cvtColor(m_LDR, cv2.COLOR_BGR2RGB) / 255.),
                                                                       (cv2.cvtColor(l_LDR, cv2.COLOR_BGR2RGB) / 255.),
                                                                       (cv2.cvtColor(s_LDR_gt, cv2.COLOR_BGR2RGB) / 255.),
                                                                       (cv2.cvtColor(m_LDR_gt, cv2.COLOR_BGR2RGB) / 255.),
                                                                       (cv2.cvtColor(l_LDR_gt, cv2.COLOR_BGR2RGB) / 255.),
                                                                        cv2.cvtColor(GT_HDR, cv2.COLOR_BGR2RGB)]
        s_gamma = 2.24
        if random.random() < 0.3:
            s_gamma += (random.random() * 0.2 - 0.1)
        
        s_HDR = ev_alignment(s_LDR, -2, s_gamma)
        m_HDR = ev_alignment(m_LDR, 0, s_gamma)
        l_HDR = ev_alignment(l_LDR, 2, s_gamma)

        s_HDR_gt = ev_alignment(s_LDR_gt, -2, s_gamma)
        m_HDR_gt = ev_alignment(m_LDR_gt, 0, s_gamma)
        l_HDR_gt = ev_alignment(l_LDR_gt, 2, s_gamma)
        
        # Convert numpy type to torch tensor type
        [s_LDR, m_LDR, l_LDR] = [torch.from_numpy(s_LDR.transpose(2,0,1).astype(np.float32)) * 2. - 1.,
                                 torch.from_numpy(m_LDR.transpose(2,0,1).astype(np.float32)) * 2. - 1.,
                                 torch.from_numpy(l_LDR.transpose(2,0,1).astype(np.float32)) * 2. - 1.]

        [s_HDR, m_HDR, l_HDR] = [torch.from_numpy(s_HDR.transpose(2,0,1).astype(np.float32)) * 2. - 1.,
                                 torch.from_numpy(m_HDR.transpose(2,0,1).astype(np.float32)) * 2. - 1.,
                                 torch.from_numpy(l_HDR.transpose(2,0,1).astype(np.float32)) * 2. - 1.]

        [s_LDR_gt, m_LDR_gt, l_LDR_gt] = [torch.from_numpy(s_LDR_gt.transpose(2,0,1).astype(np.float32)) * 2. - 1.,
                                          torch.from_numpy(m_LDR_gt.transpose(2,0,1).astype(np.float32)) * 2. - 1.,
                                          torch.from_numpy(l_LDR_gt.transpose(2,0,1).astype(np.float32)) * 2. - 1.]

        [s_HDR_gt, m_HDR_gt, l_HDR_gt] = [torch.from_numpy(s_HDR_gt.transpose(2,0,1).astype(np.float32)) * 2. - 1.,
                                          torch.from_numpy(m_HDR_gt.transpose(2,0,1).astype(np.float32)) * 2. - 1.,
                                          torch.from_numpy(l_HDR_gt.transpose(2,0,1).astype(np.float32)) * 2. - 1.]

        [GT_HDR] = [torch.from_numpy(GT_HDR.transpose(2,0,1).astype(np.float32)) * 2. - 1.]
        
        
        return {'s_LDR': s_LDR, 'm_LDR': m_LDR, 'l_LDR': l_LDR,
                's_HDR': s_HDR, 'm_HDR': m_HDR, 'l_HDR': l_HDR,
                's_LDR_gt': s_LDR_gt, 'm_LDR_gt': m_LDR_gt, 'l_LDR_gt': l_LDR_gt,
                's_HDR_gt': s_HDR_gt, 'm_HDR_gt': m_HDR_gt, 'l_HDR_gt': l_HDR_gt,
                'GT_HDR': GT_HDR}

    def __len__(self):
        return self.img_num

    def name(self):
        return 'Image HDR Dataset'