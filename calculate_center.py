# -*- coding: utf-8 -*-
# @Time        : 2021/1/23 16:17
# @Author      : ssxy00
# @File        : calculate_center.py
# @Description : TODO: calculate_center 的部分用 for 循环在算，所以效率很低，改成用 tensor 的写法

import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataset import WebFaceDataset
from model import ResnetFRModel

def calculate_center(model, train_dataset, args, device):
    centers = torch.zeros((train_dataset.n_class, 512), device=device)
    counts = torch.zeros((train_dataset.n_class), dtype=torch.int, device=device)
    with torch.no_grad():
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        for data_idx, data in enumerate(tqdm(train_dataloader)):
            data = {key: data[key].to(device) for key in data}
            batch_image1_features = model(data["images"], only_feature=True)
            for feature, label in zip(batch_image1_features, data["labels"]):
                centers[label, :] += feature
                counts[label] += 1

    centers = centers / counts.unsqueeze(-1)
    return centers

def main(args):
    # config
    device = torch.device(args.device)
    # read name list
    label2name = {}
    name2label = {}
    with open(args.name_list_file) as fin:
        label = 0
        for line in fin:
            _, name = line.split()
            label2name[label] = name
            name2label[name] = label
            label += 1

    # load data
    train_dataset = WebFaceDataset(list_file=args.train_list_file, label2name=label2name, name2label=name2label)

    # load checkpoint
    model = ResnetFRModel(n_class=len(label2name), resnet_pretrain=False)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.to(device)
    model.eval()

    # calculate center
    centers = calculate_center(model=model, train_dataset=train_dataset, args=args, device=device)
    torch.save(centers, args.center_path)


def cli_main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--train_list_file",
                        default="/home1/sxy/datasets/face_recognition/CASIA-WebFace-new/train_list.jsonl",
                        help="each line records the metadata of an image belonging to the training set")
    parser.add_argument("--name_list_file", default="/home1/sxy/datasets/face_recognition/CASIA-WebFace-new/names.txt",
                        help="this file records the name of each identity")
    # ckpt
    parser.add_argument("--checkpoint_path", type=str,
                        default="/home1/sxy/face_recognition/checkpoints/ce_only/with_pretrain/final/checkpoint16.pt",
                        help="path to load model checkpoint")
    # save
    parser.add_argument("--center_path", type=str, default="/home1/sxy/datasets/face_recognition/CASIA-WebFace-new/center.pt",
                        help="path to save calculated centers")

    # basic config
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help="cpu or cuda")
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')


    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
