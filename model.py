# -*- coding: utf-8 -*-
# @Time        : 2021/1/13 16:04
# @Author      : ssxy00
# @File        : model.py
# @Description :

import torch
import torch.nn as nn
import torchvision.models as models


class ResnetFRModel(nn.Module):
    # TODO 这个写法会引入额外的参数 fc 层
    def __init__(self, n_class, resnet_pretrain):
        super(ResnetFRModel, self).__init__()
        self.backbone = models.resnet18(pretrained=resnet_pretrain)
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
