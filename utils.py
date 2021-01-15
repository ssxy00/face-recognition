# -*- coding: utf-8 -*-
# @Time        : 2021/1/13 16:04
# @Author      : ssxy00
# @File        : utils.py
# @Description :

import random
import torch
from torchvision import transforms


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def train_transform_f():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )


