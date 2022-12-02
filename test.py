import time
from options.test_options import TestOptions
from data.custom_dataset_data_loader import CreateDataLoader
from model.custom_model import HDRNet
import os
import torch
import numpy as np
import cv2

######## for controllable results ########
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

opt = TestOptions().parse()
img_test_loader_static, img_test_loader_dynamic, img_test_loader_sensetime, img_test_loader_mix = CreateDataLoader(opt)

model = HDRNet()
model.initialize(opt)

# model.img_testHDR(img_test_loader_static, int(opt.which_iter), 'static')
# model.img_testHDR_full(img_test_loader_dynamic, int(opt.which_iter), 'dynamic')
model.img_testHDR(img_test_loader_mix, int(opt.which_iter), 'mix')
# model.img_testHDR_sensetime(img_test_loader_sensetime, int(opt.which_iter))