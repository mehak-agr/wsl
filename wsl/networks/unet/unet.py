# from torch import nn
# import torchvision
# resnet = torchvision.models.resnet.resnet50(pretrained=True)
# import torch
# import torch.nn.functional as F


# class _ASPPModule(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size, padding, dilation):
#         super(_ASPPModule, self).__init__()
#         self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
#                                      stride=1, padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU()

#         self._init_weight()

#     def forward(self, x):
#         x = self.atrous_conv(x)
#         x = self.bn(x)

#         return self.relu(x)

#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()


# class ASPP(nn.Module):
#     def __init__(self, inplanes=512, mid_c=256, dilations=[1, 6, 12, 18]):
#         super(ASPP, self).__init__()
#         self.aspp1 = _ASPPModule(inplanes, mid_c, 1, padding=0, dilation=dilations[0])
#         self.aspp2 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[1], dilation=dilations[1])
#         self.aspp3 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[2], dilation=dilations[2])
#         self.aspp4 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[3], dilation=dilations[3])

#         self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
#                                              nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
#                                              nn.BatchNorm2d(mid_c),
#                                              nn.ReLU())
#         self.conv1 = nn.Conv2d(mid_c * 5, mid_c, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(mid_c)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self._init_weight()

#     def forward(self, x):
#         x1 = self.aspp1(x)
#         x2 = self.aspp2(x)
#         x3 = self.aspp3(x)
#         x4 = self.aspp4(x)
#         x5 = self.global_avg_pool(x)
#         x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
#         x = torch.cat((x1, x2, x3, x4, x5), dim=1)

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         return self.dropout(x)

#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()


# class Unet(nn.Module):
#     def __init__(self):
#         super(Unet, self).__init__()
#         self.down = True
#         self.basemodel = resnet34(True)
#         self.planes = [256 // 4, 512 // 4, 1024 // 4, 2048 // 4]
#         self.center = ASPP(self.planes[3], self.planes[2])
#         self.fc_op = nn.Sequential(
#             nn.Conv2d(self.planes[2], 64, kernel_size=1),
#             nn.AdaptiveAvgPool2d(1))

#         self.fc = nn.Linear(64, 1)
#         self.UP4 = UpBlock(self.planes[2], 64, 64)
#         self.UP3 = UpBlock(self.planes[2] + 64, 64, 64)
#         self.UP2 = UpBlock(self.planes[1] + 64, 64, 64)
#         self.UP1 = UpBlock(self.planes[0] + 64, 64, 64)
#         self.final = nn.Sequential(
#             nn.Conv2d(64 * 4, self.planes[0] // 2, kernel_size=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(self.planes[0] // 2),

#             nn.Conv2d(self.planes[0] // 2, self.planes[0] // 2, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(self.planes[0] // 2),

#             nn.UpsamplingBilinear2d(size=(SIZE, SIZE)),

#             nn.Conv2d(self.planes[0] // 2, NUM_CLASSES, kernel_size=1)
#         )

#     def forward(self, x):
#         x1, x2, x3, x4 = self.basemodel(x)
#         if self.down:
#             x1 = self.down1(x1)
#             x2 = self.down2(x2)
#             x3 = self.down3(x3)
#             x4 = self.down4(x4)
#         x4 = self.center(x4)
#         fc_feat = self.fc_op(x4)
#         fc = fc_feat.view(fc_feat.size(0), -1)
#         fc = self.fc(fc)
#         x4 = self.UP4(x4)
#         x3 = self.UP3(torch.cat([x3, x4], 1))
#         x2 = self.UP2(torch.cat([x2, x3], 1))
#         x1 = self.UP1(torch.cat([x1, x2], 1))
#         h, w = x1.size()[2:]
#         x = torch.cat(
#             [
#                 # F.upsample_bilinear(fc_feat, size=(h, w)),
#                 F.upsample_bilinear(x4, size=(h, w)),
#                 F.upsample_bilinear(x3, size=(h, w)),
#                 F.upsample_bilinear(x2, size=(h, w)),
#                 x1
#             ],
#             1
#         )
#         return self.final(x), fc

""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# # Adapted from https://discuss.pytorch.org/t/unet-implementation/426

# import torch
# from torch import nn
# import torch.nn.functional as F


# class UNet(nn.Module):
#     def __init__(self, in_channels=3, n_classes=1, depth=5, wf=6, padding=False,
#                  batch_norm=False, up_mode='upconv'):
#         """
#         Implementation of
#         U-Net: Convolutional Networks for Biomedical Image Segmentation
#         (Ronneberger et al., 2015)
#         https://arxiv.org/abs/1505.04597
#         Using the default arguments will yield the exact version used
#         in the original paper
#         Args:
#             in_channels (int): number of input channels
#             n_classes (int): number of output channels
#             depth (int): depth of the network
#             wf (int): number of filters in the first layer is 2**wf
#             padding (bool): if True, apply padding such that the input shape
#                             is the same as the output.
#                             This may introduce artifacts
#             batch_norm (bool): Use BatchNorm after layers with an
#                                activation function
#             up_mode (str): one of 'upconv' or 'upsample'.
#                            'upconv' will use transposed convolutions for
#                            learned upsampling.
#                            'upsample' will use bilinear upsampling.
#         """
#         super(UNet, self).__init__()
#         assert up_mode in ('upconv', 'upsample')
#         self.padding = padding
#         self.depth = depth
#         prev_channels = in_channels
#         self.down_path = nn.ModuleList()
#         for i in range(depth):
#             self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
#                                                 padding, batch_norm))
#             prev_channels = 2**(wf+i)

#         self.up_path = nn.ModuleList()
#         for i in reversed(range(depth -1)):
#             self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
#                                             padding, batch_norm))
#             prev_channels = 2**(wf+i)

#         self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

#     def forward(self, x):
#         blocks = []
#         for i, down in enumerate(self.down_path):
#             x = down(x)
#             if i != len(self.down_path)-1:
#                 blocks.append(x)
#                 x = F.avg_pool2d(x, 2)

#         for i, up in enumerate(self.up_path):
#             x = up(x, blocks[-i-1])

#         return self.last(x)


# class UNetConvBlock(nn.Module):
#     def __init__(self, in_size, out_size, padding, batch_norm):
#         super(UNetConvBlock, self).__init__()
#         block = []

#         block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
#                                padding=int(padding)))
#         block.append(nn.ReLU())
#         if batch_norm:
#             block.append(nn.BatchNorm2d(out_size))

#         block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
#                                padding=int(padding)))
#         block.append(nn.ReLU())
#         if batch_norm:
#             block.append(nn.BatchNorm2d(out_size))

#         self.block = nn.Sequential(*block)

#     def forward(self, x):
#         out = self.block(x)
#         return out


# class UNetUpBlock(nn.Module):
#     def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
#         super(UNetUpBlock, self).__init__()
#         if up_mode == 'upconv':
#             self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
#                                          stride=2)
#         elif up_mode == 'upsample':
#             self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
#                                     nn.Conv2d(in_size, out_size, kernel_size=1))

#         self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

#     def center_crop(self, layer, target_size):
#         _, _, layer_height, layer_width = layer.size()
#         diff_y = (layer_height - target_size[0]) // 2
#         diff_x = (layer_width - target_size[1]) // 2
#         return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

#     def forward(self, x, bridge):
#         up = self.up(x)
#         crop1 = self.center_crop(bridge, up.shape[2:])
#         out = torch.cat([up, crop1], 1)
#         out = self.conv_block(out)

#         return out
