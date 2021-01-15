# -*- coding: utf-8 -*-
# @Time        : 2021/1/13 16:04
# @Author      : ssxy00
# @File        : dataset.py
# @Description :


import os
import jsonlines
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils import train_transform_f


class WebFaceDataset(Dataset):
    def __init__(self, list_file, label2name, name2label):
        super(WebFaceDataset, self).__init__()
        self.label2name = label2name
        self.name2label = name2label
        self.n_class = len(self.label2name)
        # load image infos
        self.image_infos = []
        with open(list_file) as fin:
            for image_info in jsonlines.Reader(fin):
                self.image_infos.append(image_info)
        # transform
        self.transforms = train_transform_f()

    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, idx):
        image_info = self.image_infos[idx]
        image, label = self.load_image(image_info)
        return {"images": image, "labels": label}


    def load_image(self, image_info):
        image = Image.open(image_info["cropped_file_path"])
        image = self.transforms(image)
        label = self.name2label[image_info["identity_name"]]
        return image, label
