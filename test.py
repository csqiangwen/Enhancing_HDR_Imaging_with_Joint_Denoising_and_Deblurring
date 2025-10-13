import time
from options.test_options import TestOptions
from data.custom_dataset_data_loader import CreateDataLoader
from model.custom_model import CustomNet
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
img_test_loader = CreateDataLoader(opt)

model = CustomNet()
model.initialize(opt)

if opt.isCustomData:
    model.img_testHDR_full_linear_merge(img_test_loader, int(opt.which_iter), opt.customName)
else:
    model.img_testHDR_full_linear_merge(img_test_loader, int(opt.which_iter), 'GT')
