import math
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import cv2

# Input range [-1, 1]
def compute_psnr(img1, img2):
    img1 = np.clip(img1, -1, 1)
    img1 = img1 / 2 + 0.5
    img2 = img2 / 2 + 0.5
    return peak_signal_noise_ratio(img1, img2, data_range=img2.max() - img2.min())

def compute_ssim(img1, img2):
    img1 = np.clip(img1, -1, 1)
    img1 = img1 / 2 + 0.5
    img2 = img2 / 2 + 0.5
    return structural_similarity(img1, img2, data_range=img2.max() - img2.min(), multichannel=True)
