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

def ev_alignment(img, expo, gamma):
    return ((img ** gamma) * 2.0**(-1*expo))**(1/gamma)

def load_image_by_keyword(base_path: Path, keyword: str, extensions=['.png', '.JPEG', '.jpeg', '.JPG']):
    # Search for files in base_path that contain the keyword and have one of the allowed extensions
    for ext in extensions:
        # Use glob pattern to find files containing the keyword and with the extension
        pattern = f'*{keyword}*{ext}'
        files = list(base_path.glob(pattern))
        if files:
            # Load the first matching file
            img = cv2.imread(str(files[0]))
            if img is not None:
                return img
    return None

## Dataloader for HDR dataset
class HDRDataset(data.Dataset):
    def __init__(self, opt):
        super(HDRDataset,self).__init__()
        self.opt = opt
        self.folder_path = Path(self.opt.hdr_dararoot)
        self.folder_list = natsorted(os.listdir(self.folder_path))

        self.img_num = len(self.folder_list)

    def mod_crop(self, img_list):
        for i, img in enumerate(img_list):
            # img_list[i] = util.modcrop(self._center_crop(img), 16)
            # img_list[i] = self._center_crop(img)
            img_list[i] = util.modcrop(img, 16)
            pass
        
        return img_list

    def _center_crop(self, x):
        crop_h, crop_w = (2800, 4800)
        h, w = x.shape[:2]
        j = int(round((h - crop_h) / 2.))
        i = int(round((w - crop_w) / 2.))
        x = x[max(0, j):min(h, j + crop_h), max(0, i):min(w, i + crop_w), :]
        # if x.shape[:2] != (crop_h, crop_w):
        # x = cv2.resize(x, (crop_w//2, crop_h//2), interpolation=cv2.INTER_LINEAR)
        return x


    def __getitem__(self, index):

        base_folder = self.folder_path / self.folder_list[index]
        input_folder = base_folder / 'input'

        if input_folder.exists() and input_folder.is_dir():
            img_path = input_folder
        else:
            img_path = base_folder
        # Read images
        s_LDR = load_image_by_keyword(img_path, 'short')
        m_LDR = load_image_by_keyword(img_path, 'medium')
        l_LDR = load_image_by_keyword(img_path, 'long')
        
        [s_LDR, m_LDR, l_LDR] = self.mod_crop([s_LDR, m_LDR, l_LDR])
        # Convert BGR to BGR color space and normalize values in [-1, 1]
        [s_LDR, m_LDR, l_LDR] = [(cv2.cvtColor(s_LDR, cv2.COLOR_BGR2RGB) / 255.),
                                 (cv2.cvtColor(m_LDR, cv2.COLOR_BGR2RGB) / 255.) ,
                                 (cv2.cvtColor(l_LDR, cv2.COLOR_BGR2RGB) / 255.)]

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

        
        
        return {'s_LDR': s_LDR, 'm_LDR': m_LDR, 'l_LDR': l_LDR,
                's_HDR': s_HDR, 'm_HDR': m_HDR, 'l_HDR': l_HDR}

    def __len__(self):
        return self.img_num

    def name(self):
        return 'Image HDR Dataset'