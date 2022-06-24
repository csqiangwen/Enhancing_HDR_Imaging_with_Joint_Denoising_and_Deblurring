import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DataParallel

##################################################

def trainable(net, trainable):
    for para in net.parameters():
        para.requires_grad = trainable

#Initialize VGG16 with pretrained weight on ImageNet
def vgg_init():
    vgg_model = torchvision.models.vgg16(pretrained = True).cuda()
    trainable(vgg_model, False)
    return vgg_model

#Extract features from internal layers for perceptual loss
class vgg(nn.Module):
    def __init__(self, vgg_model):
        super(vgg, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2",
            '6': "relu2_1",
            '8': "relu2_2"
        }

    def forward(self, x):
        outputs = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outputs.append(x)
        return outputs

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()        
        self.vgg = DataParallel(vgg(vgg_init()).cuda())
        self.criterion = nn.MSELoss()
        self.weights = [1.0/16, 1.0/8, 1.0/4, 1.0]
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        # x = self.normalize((x+1)/2)
        # y = self.normalize((y+1)/2)
        x_outputs, y_outputs = self.vgg(x), self.vgg(y)
        loss = 0
        i = 0
        for x_output, y_output in zip(x_outputs, y_outputs):
            loss += self.criterion(x_output, y_output)*self.weights[i]
            i += 1
        return loss
