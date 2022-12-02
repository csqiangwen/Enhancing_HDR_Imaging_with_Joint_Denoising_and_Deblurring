import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math
import torch.nn.functional as F


class ImageGradient(nn.Module):
    def __init__(self):
        super(ImageGradient, self).__init__()

        a = np.array([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
                      [[[1, 0, -1], [2, 0, -2], [1, 0, -1]]],
                      [[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]])
        conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        conv1.weight = nn.Parameter(torch.from_numpy(a).float(), requires_grad=False)


        b = np.array([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
                      [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]],
                      [[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
        conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        conv2.weight = nn.Parameter(torch.from_numpy(b).float(), requires_grad=False)

        self.conv1 = conv1
        self.conv2 = conv2

    def forward(self, input):
        G_x = self.conv1(input)
        G_y = self.conv2(input)
        
        return torch.cat([G_x, G_y], dim=1)

class ImageSmoothing(nn.Module):
    def __init__(self):
        super(ImageSmoothing, self).__init__()

        a = np.array([[[[1/16., 2/16., 1/16.], [2/16., 4/16., 2/16.], [1/16., 2/16., 1/16.]]],
                      [[[1/16., 2/16., 1/16.], [2/16., 4/16., 2/16.], [1/16., 2/16., 1/16.]]],
                      [[[1/16., 2/16., 1/16.], [2/16., 4/16., 2/16.], [1/16., 2/16., 1/16.]]]])
        conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
        conv1.weight = nn.Parameter(torch.from_numpy(a).float(), requires_grad=False)

        self.conv1 = conv1

    def forward(self, input):
        smoothed_input = self.conv1(input)
        
        return smoothed_input

    
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


def tanh_norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    bounded_hdr = torch.tanh(hdr_image / norm_value)
    return mu_tonemap(bounded_hdr, mu)


# input is [-1, 1]
def mu_tonemap(hdr_image, mu=5000):
    return torch.log(1 + mu * (hdr_image + 1.) / 2.) / torch.log(1 + torch.tensor(mu)) * 2. - 1


class mu_loss(object):
    def __init__(self, gamma=2.24, percentile=99):
        self.gamma = gamma
        self.percentile = percentile

    def __call__(self, pred, label):
        hdr_linear_ref = pred ** self.gamma
        hdr_linear_res = label ** self.gamma
        norm_perc = np.percentile(hdr_linear_ref.data.cpu().numpy().astype(np.float32), self.percentile)
        mu_pred = tanh_norm_mu_tonemap(hdr_linear_ref, norm_perc)
        mu_label = tanh_norm_mu_tonemap(hdr_linear_res, norm_perc)
        return nn.L1Loss()(mu_pred, mu_label)

     