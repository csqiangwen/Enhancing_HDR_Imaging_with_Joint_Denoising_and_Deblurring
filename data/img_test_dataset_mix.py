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
        self.folder_path = root_path / 'test_mix' / 'dynamic'
        self.folder_list = natsorted(os.listdir(self.folder_path))

        self.img_num = len(self.folder_list)

    def mod_crop(self, img_list):
        for i, img in enumerate(img_list):
            img_list[i] = util.modcrop(self._center_crop(img), 128)
            # img_list[i] = self._center_crop(img)
        
        return img_list

    def _center_crop(self, x):
        crop_h, crop_w = (2800, 4800)
        h, w = x.shape[:2]
        j = int(round((h - crop_h) / 2.))
        i = int(round((w - crop_w) / 2.))
        x = x[max(0, j):min(h, j + crop_h), max(0, i):min(w, i + crop_w), :]
        # if x.shape[:2] != (crop_h, crop_w):
        # x = cv2.resize(x, (crop_w, crop_h))
        return x


    def __getitem__(self, index):

        img_path = self.folder_path / self.folder_list[index]
        # Read images
        [s_LDR, m_LDR, l_LDR, GT_HDR] = [cv2.imread(str(img_path / 'input' / 'short.png')),
                                         cv2.imread(str(img_path / 'input' / 'medium.png')),
                                         cv2.imread(str(img_path / 'input' / 'long.png')),
                                         cv2.imread(str(img_path / 'gt' / 'HDR_p_norm.hdr'), flags=cv2.IMREAD_ANYDEPTH)]
        
        [s_LDR, m_LDR, l_LDR, GT_HDR] = self.mod_crop([s_LDR, m_LDR, l_LDR, GT_HDR])
        # Convert BGR to BGR color space and normalize values in [-1, 1]
        [s_LDR, m_LDR, l_LDR, GT_HDR] = [(cv2.cvtColor(s_LDR, cv2.COLOR_BGR2RGB) / 255.),
                                         (cv2.cvtColor(m_LDR, cv2.COLOR_BGR2RGB) / 255.) ,
                                         (cv2.cvtColor(l_LDR, cv2.COLOR_BGR2RGB) / 255.) ,
                                          cv2.cvtColor(GT_HDR, cv2.COLOR_BGR2RGB)]

        s_gamma = 2.24
        
        s_HDR = ev_alignment(s_LDR, -2, s_gamma)
        m_HDR = ev_alignment(m_LDR, 0, s_gamma)
        l_HDR = ev_alignment(l_LDR, 2, s_gamma)
        
        # Convert numpy type to torch tensor type
        [s_LDR, m_LDR, l_LDR] = [torch.from_numpy(s_LDR.transpose(2,0,1).astype(np.float32)) * 2. - 1.,
                                 torch.from_numpy(m_LDR.transpose(2,0,1).astype(np.float32)) * 2. - 1.,
                                 torch.from_numpy(l_LDR.transpose(2,0,1).astype(np.float32)) * 2. - 1.]

        [s_HDR, m_HDR, l_HDR] = [torch.from_numpy(s_HDR.transpose(2,0,1).astype(np.float32)) * 2. - 1.,
                                 torch.from_numpy(m_HDR.transpose(2,0,1).astype(np.float32)) * 2. - 1.,
                                 torch.from_numpy(l_HDR.transpose(2,0,1).astype(np.float32)) * 2. - 1.]

        [GT_HDR] = [torch.from_numpy(GT_HDR.transpose(2,0,1).astype(np.float32)) * 2. - 1.]
        
        
        return {'s_LDR': s_LDR, 'm_LDR': m_LDR, 'l_LDR': l_LDR,
                's_HDR': s_HDR, 'm_HDR': m_HDR, 'l_HDR': l_HDR,
                'GT_HDR': GT_HDR}

    def __len__(self):
        return self.img_num

    def name(self):
        return 'Image HDR Dataset'