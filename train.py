# -*- coding: utf-8 -*-
# @Time        : 2021/1/13 10:42
# @Author      : ssxy00
# @File        : train.py
# @Description :

import os
import argparse

import torch

from utils import set_seed
from dataset import WebFaceDataset
from model import FRModel, ResnetFRModel
from trainer import FRTrainer


def main(args):
    # config
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    set_seed(args.seed)
    device = torch.device(args.device)

    # data
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
    # load training dataset
    train_dataset = WebFaceDataset(list_file=args.train_list_file, label2name=label2name, name2label=name2label)
    # load validation dataset
    valid_dataset = WebFaceDataset(list_file=args.valid_list_file, label2name=label2name, name2label=name2label)

    # model
    # model = FRModel(n_class=len(label2name))
    model = ResnetFRModel(n_class=len(label2name))

    # trainer
    trainer = FRTrainer(train_dataset=train_dataset, valid_dataset=valid_dataset, model=model, args=args)

    trainer.train()


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list_file",
                        default="/home1/sxy/datasets/face_recognition/CASIA-WebFace/train_list.jsonl",
                        help="each line records the metadata of an image belonging to the training set")
    parser.add_argument("--valid_list_file",
                        default="/home1/sxy/datasets/face_recognition/CASIA-WebFace/valid_list.jsonl",
                        help="each line records the metadata of an image belonging to the validation set")
    parser.add_argument("--name_list_file", default="/home1/sxy/datasets/face_recognition/CASIA-WebFace/names.txt",
                        help="this file records the name of each identity")

    parser.add_argument("--last_ckpt", default=0, type=int, help="last checkpoint, if 0, train from scratch")
    parser.add_argument("--save_model_dir", type=str, default="/home1/sxy/face_recognition/checkpoints/ce_only",
                        help="path to save model checkpoint")
    parser.add_argument("--log_dir", type=str, default="/home1/sxy/face_recognition/logs/ce_only",
                        help="path to save log")

    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help="cpu or cuda")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument("--n_epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--save_interval", default=1)

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
