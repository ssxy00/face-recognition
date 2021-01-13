# -*- coding: utf-8 -*-
# @Time        : 2021/1/13 20:45
# @Author      : ssxy00
# @File        : random_split.py
# @Description :

"""
Pipeline: 4
将包含所有图片经过 detect 和 align 操作后信息的 .jsonl 文件随机分成 training set 和 validation set
"""

import argparse
import jsonlines
import random


def main(args):
    image_infos = []
    with open(args.cropped_image_list_file) as fin:
        for image_info in jsonlines.Reader(fin):
            image_infos.append(image_info)

    random.shuffle(image_infos)
    valid_size = int(len(image_infos) * args.valid_ratio)

    with jsonlines.open(args.train_list_file, 'w') as fout:
        for image_info in image_infos[:-valid_size]:
            fout.write(image_info)
    with jsonlines.open(args.valid_list_file, 'w') as fout:
        for image_info in image_infos[-valid_size:]:
            fout.write(image_info)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cropped_image_list_file",
                        default="/home1/sxy/datasets/face_recognition/CASIA-WebFace/cropped_image_list.jsonl",
                        help="input file, each line records the metadata of an image")
    parser.add_argument("--train_list_file",
                        default="/home1/sxy/datasets/face_recognition/CASIA-WebFace/train_list.jsonl",
                        help="output file, each line records the metadata of an image belonging to the training set")
    parser.add_argument("--valid_list_file",
                        default="/home1/sxy/datasets/face_recognition/CASIA-WebFace/valid_list.jsonl",
                        help="output file, each line records the metadata of an image belonging to the validation set")
    parser.add_argument("--valid_ratio", type=float, default=0.1, help="proportion of validation set to total data")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
