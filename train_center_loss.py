# -*- coding: utf-8 -*-
# @Time        : 2021/1/23 16:09
# @Author      : ssxy00
# @File        : train_center_loss.py
# @Description :

import os
import argparse

import torch

from utils import set_seed
from dataset import WebFaceDataset
from model import ResnetFRModel
from center_loss_trainer import FRTrainer


def main(args):
    # config
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    set_seed(args.seed)

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
    model = ResnetFRModel(n_class=len(label2name), resnet_pretrain=False, center_loss=True)
    model.load_state_dict(torch.load(args.checkpoint_path))

    # trainer
    trainer = FRTrainer(train_dataset=train_dataset, valid_dataset=valid_dataset, model=model, args=args)

    trainer.train()


def cli_main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--train_list_file",
                        default="/home1/sxy/datasets/face_recognition/CASIA-WebFace-new/train_list.jsonl",
                        help="each line records the metadata of an image belonging to the training set")
    parser.add_argument("--valid_list_file",
                        default="/home1/sxy/datasets/face_recognition/CASIA-WebFace-new/valid_list.jsonl",
                        help="each line records the metadata of an image belonging to the validation set")
    parser.add_argument("--name_list_file", default="/home1/sxy/datasets/face_recognition/CASIA-WebFace-new/names.txt",
                        help="this file records the name of each identity")
    parser.add_argument("--center_path", type=str,
                        default="/home1/sxy/datasets/face_recognition/CASIA-WebFace-new/center.pt",
                        help="path to load calculated centers")
    # save
    parser.add_argument("--save_model_dir", type=str, default="/home1/sxy/face_recognition/checkpoints/tmp",
                        help="path to save model checkpoint")
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default="/home1/sxy/face_recognition/logs/tmp",
                        help="path to save log")
    # initial ckpt
    parser.add_argument("--checkpoint_path", type=str,
                        default="/home1/sxy/face_recognition/checkpoints/ce_only/with_pretrain/final/checkpoint16.pt",
                        help="path to load model checkpoint")
    # basic config
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help="cpu or cuda")
    # training
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument("--n_epochs", default=10, type=int, help="number of training epochs")
    # lr
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--lr_schedule", default="none", type=str, help="multi_step or none")
    parser.add_argument("--gamma", default=0.1, type=float, help="lr decay factor")
    # center loss
    parser.add_argument("--lambda_factor", default=1, type=float)
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--ce_factor", default=1, type=float)


    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
