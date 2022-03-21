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

## Dataloader for RainDrop dataset
class HDRDataset(data.Dataset):
    def __init__(self, opt):
        super(HDRDataset,self).__init__()
        self.opt = opt
        self.folder_path = os.path.join(self.opt.hdr_dararoot, 'Test', 'EXTRA')
        self.folder_list = natsorted(os.listdir(self.folder_path))

        self.img_num = len(self.folder_list)

    def mod_crop(self, img_list):
        for i, img in enumerate(img_list):
            # img_list[i] = util.modcrop(self._center_crop(img), 8)
            img_list[i] = self._center_crop(img)
        
        return img_list

    def _center_crop(self, x):
        crop_h, crop_w = (960, 1440)
        h, w = x.shape[:2]
        j = int(round((h - crop_h) / 2.))
        i = int(round((w - crop_w) / 2.))
        x = x[max(0, j):min(h, j + crop_h), max(0, i):min(w, i + crop_w), :]
        if x.shape[:2] != (crop_h, crop_w):
            x = cv2.resize(x, (crop_w, crop_h))
        return x


    def __getitem__(self, index):

        img_path = os.path.join(self.folder_path, self.folder_list[index])
        img_list = natsorted(os.listdir(img_path))
        # Read images
        [s_LDR, m_LDR, l_LDR, GT_HDR] = [cv2.imread(os.path.join(img_path, img_list[0])),
                                         cv2.imread(os.path.join(img_path, img_list[1])),
                                         cv2.imread(os.path.join(img_path, img_list[2])),
                                         cv2.imread(os.path.join(img_path, img_list[3]), flags=cv2.IMREAD_ANYDEPTH)]
        [s_LDR, m_LDR, l_LDR, GT_HDR] = self.mod_crop([s_LDR, m_LDR, l_LDR, GT_HDR])
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

        exps = hdr_util._get_exps(os.path.join(img_path, img_list[4]))

        [s_HDR, m_HDR, l_HDR] = [hdr_util.ldr2hdr(s_LDR, exps[0]),
                                 hdr_util.ldr2hdr(m_LDR, exps[1]),
                                 hdr_util.ldr2hdr(l_LDR, exps[2])]

        return {'s_LDR': s_LDR, 'm_LDR': m_LDR, 'l_LDR': l_LDR, 'GT_HDR': GT_HDR, 'dark_c': dark_c,
                's_HDR': s_HDR, 'm_HDR': m_HDR, 'l_HDR': l_HDR, 'bright_c': bright_c,}

    def __len__(self):
        return self.img_num

    def name(self):
        return 'Image HDR Dataset'