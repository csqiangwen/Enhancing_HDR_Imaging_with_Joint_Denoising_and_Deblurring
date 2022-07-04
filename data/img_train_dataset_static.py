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

## Dataloader for RainDrop dataset
class HDRDataset(data.Dataset):
    def __init__(self, opt):
        super(HDRDataset,self).__init__()
        self.opt = opt
        root_path = Path(self.opt.hdr_dararoot)
        self.folder_path = root_path / 'train_ps' / 'static'
        self.folder_list = natsorted(os.listdir(self.folder_path))

        self.img_num = len(self.folder_list)

    def random_crop_flip(self, img_list, opt):
        if random.random()>0.3:
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


    def __getitem__(self, index):

        img_path = self.folder_path / self.folder_list[index]
        img_list = natsorted(os.listdir(img_path))
        # Read images
        [s_LDR, m_LDR, l_LDR, GT_HDR] = [cv2.imread(str(img_path / 'input' / 'short.tif')),
                                         cv2.imread(str(img_path / 'input' / 'medium.tif')),
                                         cv2.imread(str(img_path / 'input' / 'long.tif')),
                                         cv2.imread(str(img_path / 'gt' / 'HDR_norm.hdr'), flags=cv2.IMREAD_ANYDEPTH)]
                                         
        [s_LDR, m_LDR, l_LDR, GT_HDR] = self.random_crop_flip([s_LDR, m_LDR, l_LDR, GT_HDR], self.opt)
        
        # Convert BGR to BGR color space and normalize values in [-1, 1]
        [s_LDR, m_LDR, l_LDR, GT_HDR] = [cv2.cvtColor(s_LDR, cv2.COLOR_BGR2RGB) / 255. * 2. - 1.,
                                         cv2.cvtColor(m_LDR, cv2.COLOR_BGR2RGB) / 255. * 2. - 1.,
                                         cv2.cvtColor(l_LDR, cv2.COLOR_BGR2RGB) / 255. * 2. - 1.,
                                         cv2.cvtColor(GT_HDR, cv2.COLOR_BGR2RGB) * 2. - 1.]

        [bright_c, dark_c] = [np.max(s_LDR, axis=-1, keepdims=True), np.min(l_LDR, axis=-1, keepdims=True)]
        # Convert numpy type to torch tensor type
        [s_LDR, m_LDR, l_LDR, GT_HDR] = [torch.from_numpy(s_LDR.transpose(2,0,1).astype(np.float32)),
                                         torch.from_numpy(m_LDR.transpose(2,0,1).astype(np.float32)),
                                         torch.from_numpy(l_LDR.transpose(2,0,1).astype(np.float32)),
                                         torch.from_numpy(GT_HDR.transpose(2,0,1).astype(np.float32))]

        [bright_c, dark_c] = [torch.from_numpy(bright_c.transpose(2,0,1).astype(np.float32)),
                              torch.from_numpy(dark_c.transpose(2,0,1).astype(np.float32))]

        # exps = hdr_util._get_exps(os.path.join(img_path, img_list[4]))

        [s_HDR, m_HDR, l_HDR] = [hdr_util.ldr2hdr(s_LDR, 2**float(0)),
                                 hdr_util.ldr2hdr(m_LDR, 2**float(2)),
                                 hdr_util.ldr2hdr(l_LDR,  2**float(4))]

        return {'s_LDR': s_LDR, 'm_LDR': m_LDR, 'l_LDR': l_LDR, 'GT_HDR': GT_HDR, 'dark_c': dark_c,
                's_HDR': s_HDR, 'm_HDR': m_HDR, 'l_HDR': l_HDR, 'bright_c': bright_c,}

    def __len__(self):
        return self.img_num

    def name(self):
        return 'Image HDR Dataset'