# -*- coding: utf-8 -*-
# @Time        : 2021/1/13 16:04
# @Author      : ssxy00
# @File        : model.py
# @Description :

# https://stackoverflow.com/questions/59455386/local-fully-connected-layer-pytorch
# http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf
# https://pytorch.org/docs/stable/tensors.html?highlight=unfold#torch.Tensor.unfold


import torch
import torch.nn as nn
import torchvision.models as models


class ResnetFRModel(nn.Module):
    # TODO 这个写法会引入额外的参数 fc 层
    def __init__(self, n_class):
        super(ResnetFRModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.classifier = nn.Linear(512, n_class)

    def forward(self, x, only_feature=False):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        if only_feature:
            return x
        return self.classifier(x)


class FRModel(nn.Module):
    def __init__(self, n_class):
        super(FRModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.activation1 = nn.PReLU()
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.activation2 = nn.PReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.activation3 = nn.PReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lconv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # TODO local connected conv
        self.activation4 = nn.PReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lconv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.activation5 = nn.PReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lconv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.activation6 = nn.PReLU()
        self.fc1 = nn.Linear(21504, 256)
        self.classifier = nn.Linear(256, n_class)

    def forward(self, x):
        x = self.conv1(x)  # B, 128, 112, 96
        x = self.activation1(x)  # B, 128, 112, 96
        x = self.conv2(x)  # B, 128, 112, 96
        x = self.activation2(x)  # B, 128, 112, 96
        x = self.pool1(x)  # B, 128, 56, 48
        x = self.conv3(x)  # B, 128, 56, 48
        x = self.activation3(x)  # B, 128, 56, 48
        x = self.pool2(x)  # B, 128, 28, 24
        x = self.lconv1(x)  # B, 256, 28, 24
        x = self.activation4(x)  # B, 256, 28, 24
        x = self.pool3(x)  # B, 256, 14, 12
        x = self.lconv2(x)  # B, 256, 14, 12
        x = self.activation5(x)  # B, 256, 14, 12
        x = self.pool4(x)  # B, 256, 7, 6
        x_1 = self.lconv3(x)  # B, 256, 7, 6
        x_1 = self.activation6(x_1)  # B, 256, 7, 6
        x = torch.cat([x.flatten(1), x_1.flatten(1)], dim=-1)
        x = self.fc1(x)
        return self.classifier(x)
