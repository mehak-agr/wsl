#!/usr/bin/python3
from torch import nn
from torchvision import models
from typing import Any, Dict
from wsl.networks.medinet.pooling import ClassWisePool, WildcatPool2d


class Architecture(nn.Module):
    def __init__(self,
                 network: str,
                 depth: int,
                 wildcat: bool,
                 classes: int,
                 maps: int,
                 alpha: float,
                 k: int,
                 pretrained: bool = False,
                 get_map: bool = False):
        super(Architecture, self).__init__()

        self.wildcat = wildcat
        self.get_map = (get_map and wildcat)
        self.network = network

        if self.network == 'densenet':
            if depth == 121:
                model = models.densenet121(pretrained, progress=False)
            elif depth == 169:
                model = models.densenet169(pretrained, progress=False)
            else:
                raise ValueError('Unsupported model depth, must be one of 121, 169')
            in_ftrs = model.classifier.in_features
            pool_size = 1
            self.features = model.features

        elif self.network == 'resnet':
            if depth == 18:
                model = models.resnet18(pretrained=True, progress=False)
            elif depth == 34:
                model = models.resnet34(pretrained=True, progress=False)
            elif depth == 50:
                model = models.resnet50(pretrained=True, progress=False)
            elif depth == 101:
                model = models.resnet101(pretrained=True, progress=False)
            elif depth == 152:
                model = models.resnet152(pretrained=True, progress=False)
            else:
                raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
            in_ftrs = model.fc.in_features
            pool_size = model.avgpool.output_size
            self.features = nn.Sequential()
            self.features.add_module('conv1', model.conv1)
            self.features.add_module('bn1', model.bn1)
            self.features.add_module('relu', model.relu)
            self.features.add_module('maxpool', model.maxpool)
            self.features.add_module('layer1', model.layer1)
            self.features.add_module('layer2', model.layer2)
            self.features.add_module('layer3', model.layer3)
            self.features.add_module('layer4', model.layer4)

        elif network == 'vgg':
            if depth == 19:
                model = models.vgg19(pretrained=True, progress=False)
            else:
                raise ValueError('Unsupported model depth, must be one of 19')
            in_ftrs = model.features[34].out_channels
            pool_size = model.avgpool.output_size
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
            self.pool = nn.AdaptiveAvgPool2d(pool_size)
            self.classifier = nn.Linear(in_ftrs, classes)

    # function to extact the features for detecting ood
    # if needed build a recursive function rather than just two levels
    def register_forward_hooks(self, net: Any, hook: Any, layer_names: Dict[str, Any]):
        handles = []
        for name, layer in net._modules.items():
            if isinstance(layer, nn.Sequential):
                for sub_name, sub_layer in layer._modules.items():
                    assert(sub_name not in layer_names)
                    layer_names[sub_layer] = sub_name
                    handles.append(sub_layer.register_forward_hook(hook))
            else:
                layer_names[layer] = name
                handles.append(layer.register_forward_hook(hook))
        return handles

    def forward(self, x):
        x = self.features(x)
        if self.wildcat:
            x = self.classifier(x)
            if self.get_map:
                return x
            x = self.pool(x)
        else:
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        return x
