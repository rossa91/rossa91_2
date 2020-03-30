'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from .module.quantize import *

cfg = {
    'VGG9':  [64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
          nn.Linear(4096, 512),
          nn.BatchNorm1d(512),
          nn.ReLU(True),
          nn.Dropout(),
          nn.Linear(512, 100),
          nn.BatchNorm1d(100),
          nn.ReLU(True),
          nn.Dropout(),
          nn.Linear(100, 10))
          
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)




class QVGG(nn.Module):
    def __init__(self, vgg_name, num_bits, mixed=True, mask=None):
        super(QVGG, self).__init__()
        self.num_bits = num_bits
        self.mixed = mixed
        self.mask = mask 
        self.features = self._make_layers(cfg[vgg_name])
        if self.mixed is True :
          self.classifier = QLinear(512, 10, num_bits=self.num_bits, num_bits_weight=self.num_bits, mixed=self.mixed, mask=mask[len(mask)-1])
        else :
          self.classifier = QLinear(512, 10, num_bits=self.num_bits, num_bits_weight=self.num_bits, mixed=self.mixed, mask=None)
    def forward(self, x):
        
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
       
        if self.mask is None and self.mixed is True:
          print("Mask Error... insert the mask")
       
        layers = []
        in_channels = 3
        i = 0
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.mixed is False :
                  mask=None
                else:
                  mask=self.mask[i]

                layers += [QConv2d(in_channels, x, kernel_size=3, padding=1, num_bits=self.num_bits, num_bits_weight=self.num_bits, mixed=self.mixed, mask=mask),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                i += 1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



def test():
#    net = QVGG('VGG9', 4, mixed=False, mask=None)
    net = VGG('VGG9')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())


test()
