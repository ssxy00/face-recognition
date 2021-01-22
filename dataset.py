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

from utils import transform_f, transform_with_flip_f


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
        self.transforms = transform_f()

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

class LFWDataset(Dataset):
    def __init__(self, image_list_file, pair_list_file):
        super(LFWDataset, self).__init__()
        # load image infos
        self.meta_to_path_dict = {}
        with open(image_list_file) as fin:
            for image_info in jsonlines.Reader(fin):
                name_idx = image_info["file_path"].split('/')[-1][:-4]  # Kirk_Franklin_0001
                self.meta_to_path_dict[name_idx] = image_info["cropped_file_path"]

        self.pairs = []
        with open(pair_list_file) as fin:
            for idx, line in enumerate(fin):
                if idx == 0:
                    continue
                label = 1 - ((idx - 1) // 300) % 2
                if label:
                    name, idx1, idx2 = line.split()
                    name_idx1 = name + '_' + '0' * (4 - len(idx1)) + idx1
                    name_idx2 = name + '_' + '0' * (4 - len(idx2)) + idx2
                else:
                    name1, idx1, name2, idx2 = line.split()
                    name_idx1 = name1 + '_' + '0' * (4 - len(idx1)) + idx1
                    name_idx2 = name2 + '_' + '0' * (4 - len(idx2)) + idx2
                self.pairs.append((name_idx1, name_idx2, label))

        # transform
        self.transforms = transform_f()
        self.transforms_with_flip = transform_with_flip_f()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name_idx1, name_idx2, label = self.pairs[idx]
        image1, image1_with_flip = self.load_image(name_idx1)
        image2, image2_with_flip = self.load_image(name_idx2)
        return {"image1": image1, "image1_with_flip": image1_with_flip,
                "image2": image2, "image2_with_flip": image2_with_flip,
                "labels": label}



    def load_image(self, name_idx):
        image = Image.open(self.meta_to_path_dict[name_idx])
        return self.transforms(image), self.transforms_with_flip(image)
