import math
import numpy as np
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
import cv2

# Input range [-1, 1]
def compute_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 2.0 # input -1~1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def eval_SSIM(img1, img2):
    img1 = cv2.cvtColor(img1[:,:,::-1], cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    img2 = cv2.cvtColor(img2[:,:,::-1], cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    ssim = compare_ssim(img1, img2, multichannel=False)

    return ssim
