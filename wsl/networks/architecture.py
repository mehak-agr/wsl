#!/usr/bin/python3
from torch import nn
from torchvision import models

from wsl.networks.pooling import ClassWisePool, WildcatPool2d


class Architecture(nn.Module):
    def __init__(self, network: str,
                 depth: int,
                 wildcat: bool,
                 classes: int,
                 maps: int,
                 alpha: float,
                 k: int,
                 pretrained: bool = True):
        super(Architecture, self).__init__()

        self.wildcat = wildcat

        if network == 'densenet':
            if depth == 121:
                model = models.densenet121(pretrained)
            elif depth == 169:
                model = models.densenet169(pretrained)
            else:
                raise ValueError('Unsupported model depth, must be one of 121, 169')
            in_ftrs = model.classifier.in_features
            self.features = model.features

        elif network == 'resnet':
            if depth == 18:
                model = models.resnet18(pretrained=True)
            elif depth == 34:
                model = models.resnet34(pretrained=True)
            elif depth == 50:
                model = models.resnet50(pretrained=True)
            elif depth == 101:
                model = models.resnet101(pretrained=True)
            elif depth == 152:
                model = models.resnet152(pretrained=True)
            else:
                raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
            in_ftrs = model.fc.in_features
            self.features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4)

        elif network == 'vgg':
            if depth == 19:
                model = models.vgg19(pretrained=True)
            else:
                raise ValueError('Unsupported model depth, must be one of 19')
            in_ftrs = model.classifier[0].in_features
            self.features = model.features

        else:
            raise ValueError('Unsupported network type, must be one of densenet, resnet, vgg')

        if wildcat:
            print('making wildcat model...', end='')
            self.classifier = nn.Conv2d(in_ftrs, maps * classes, kernel_size=1, stride=1, padding=0, bias=True)
            self.pool = nn.Sequential(ClassWisePool(maps),
                                      WildcatPool2d(kmin=k, kmax=k, alpha=alpha))
        else:
            print('making baseline model...', end='')
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(in_ftrs, classes)

    def forward(self, x):
        x = self.features(x)
        if self.wildcat:
            x = self.classifier(x)
            x = self.pool(x)
        else:
            x = self.pool(x)
            x = x.view(x.size(0), x.size(1))
            x = self.classifier(x)
        return x
