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
        self.folder_path = root_path / 'sensetime' / 'test'
        self.folder_list = natsorted(os.listdir(self.folder_path))

        self.img_num = len(self.folder_list)

    def mod_crop(self, img_list):
        for i, img in enumerate(img_list):
            # img_list[i] = util.modcrop(self._center_crop(img), 8)
            img_list[i] = self._center_crop(img)
        
        return img_list

    def _center_crop(self, x):
        # crop_h, crop_w = (912, 1368)
        # crop_h, crop_w = (3648, 5472)
        crop_h, crop_w = (2880, 3840)
        h, w = x.shape[:2]
        j = int(round((h - crop_h) / 2.))
        i = int(round((w - crop_w) / 2.))
        x = x[max(0, j):min(h, j + crop_h), max(0, i):min(w, i + crop_w), :]
        # if x.shape[:2] != (crop_h, crop_w):
        #     x = cv2.resize(x, (crop_w, crop_h))
        # x = cv2.resize(x, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        return x


    def __getitem__(self, index):

        img_path = self.folder_path / self.folder_list[index]
        img_list = natsorted(os.listdir(img_path))
        # Read images
        [s_LDR, m_LDR, l_LDR] = [cv2.imread(str(img_path / 'short.png'), flags=cv2.IMREAD_UNCHANGED),
                                 cv2.imread(str(img_path / 'medium.png'), flags=cv2.IMREAD_UNCHANGED),
                                 cv2.imread(str(img_path / 'long.png'), flags=cv2.IMREAD_UNCHANGED)]
        [s_LDR, m_LDR, l_LDR] = self.mod_crop([s_LDR, m_LDR, l_LDR])
        # Convert BGR to BGR color space and normalize values in [-1, 1]
        [s_LDR, m_LDR, l_LDR] = [cv2.cvtColor(s_LDR, cv2.COLOR_BGR2RGB) / 65535. * 2. - 1.,
                                 cv2.cvtColor(m_LDR, cv2.COLOR_BGR2RGB) / 65535. * 2. - 1.,
                                 cv2.cvtColor(l_LDR, cv2.COLOR_BGR2RGB) / 65535. * 2. - 1.]
        # Convert numpy type to torch tensor type
        [s_LDR, m_LDR, l_LDR] = [torch.from_numpy(s_LDR.transpose(2,0,1).astype(np.float32)),
                                 torch.from_numpy(m_LDR.transpose(2,0,1).astype(np.float32)),
                                 torch.from_numpy(l_LDR.transpose(2,0,1).astype(np.float32))]

        return {'s_LDR': s_LDR, 'm_LDR': m_LDR, 'l_LDR': l_LDR}

    def __len__(self):
        return self.img_num

    def name(self):
        return 'Image HDR Dataset'