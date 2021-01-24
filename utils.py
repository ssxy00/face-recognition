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


def transform_f():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )


def transform_with_flip_f():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )


def get_center_delta(features, centers, targets):
    # modified from https://github.com/louis-she/center-loss.pytorch/blob/5be899d1f622d24d7de0039dc50b54ce5a6b1151/loss.py#L14
    # implementation equation (4) in the center-loss paper
    features = features.view(features.size(0), -1)
    targets, indices = torch.sort(targets)
    target_centers = centers[targets]
    features = features[indices]

    delta_centers = target_centers - features
    uni_targets, indices = torch.unique(targets, sorted=True, return_inverse=True)

    delta_centers = torch.zeros(
        uni_targets.size(0), delta_centers.size(1)
    ).to(features.device).index_add_(0, indices, delta_centers)

    targets_repeat_num = uni_targets.size()[0]
    uni_targets_repeat_num = targets.size()[0]
    targets_repeat = targets.repeat(
        targets_repeat_num).view(targets_repeat_num, -1)
    uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
        1, uni_targets_repeat_num)
    same_class_feature_count = torch.sum(
        targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

    delta_centers = delta_centers / (same_class_feature_count + 1.0)
    result = torch.zeros_like(centers)
    result[uni_targets, :] = delta_centers
    return result
