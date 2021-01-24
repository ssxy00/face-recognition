# -*- coding: utf-8 -*-
# @Time        : 2021/1/22 16:09
# @Author      : ssxy00
# @File        : evaluate_on_lfw.py
# @Description :

import os
import math
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader

from dataset import LFWDataset
from model import ResnetFRModel


def calculate_acc_with_threshold(distances, labels, threshold):
    predict_labels = distances <= threshold
    tp = ((predict_labels == 1) & (labels == 1)).sum().item()
    tn = ((predict_labels == 0) & (labels == 0)).sum().item()
    fp = ((predict_labels == 1) & (labels == 0)).sum().item()
    fn = ((predict_labels == 0) & (labels == 1)).sum().item()
    acc = (tp + tn) / labels.shape[0]
    tpr = 0 if (tp + fn == 0) else tp / (tp + fn)
    fpr = 0 if (fp + tn == 0) else fp / (fp + tn)
    return acc, tpr, fpr


def calculate_roc(features1, features2, labels, dist_f, thresholds):
    k_fold = KFold(n_splits=10, shuffle=False)
    n_pairs = np.arange(features1.shape[0])
    # calculate distance
    cos_similarities = dist_f(features1, features2)
    distances = torch.acos(cos_similarities) / math.pi

    test_accuracy_for_each_fold = []
    true_positive_rate = np.zeros((10, len(thresholds)))
    false_positive_rate = np.zeros((10, len(thresholds)))
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(n_pairs)):
        best_accuracy = 0.
        best_threshold = 0.
        # find best threshold
        for threshold_idx, threshold in enumerate(thresholds):
            fold_accuracy, _, _ = calculate_acc_with_threshold(distances[train_set], labels[train_set], threshold)
            if fold_accuracy > best_accuracy:
                best_accuracy = fold_accuracy
                best_threshold = threshold
        # eval on test set
        test_fold_accuracy, _, _ = calculate_acc_with_threshold(distances[test_set], labels[test_set], best_threshold)
        test_accuracy_for_each_fold.append(test_fold_accuracy)
        # calculate roc
        for threshold_idx, threshold in enumerate(thresholds):
            _, test_fold_tpr, test_fold_fpr = calculate_acc_with_threshold(distances[test_set], labels[test_set],
                                                                           threshold)
            true_positive_rate[fold_idx, threshold_idx] = test_fold_tpr
            false_positive_rate[fold_idx, threshold_idx] = test_fold_fpr

    return sum(test_accuracy_for_each_fold) / len(test_accuracy_for_each_fold), np.mean(true_positive_rate, 0), \
           np.mean(false_positive_rate, 0)


def main(args):
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device)

    # load pair data
    lfw_dataset = LFWDataset(image_list_file=args.image_list_file, pair_list_file=args.pair_list_file)
    lfw_dataloader = DataLoader(lfw_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # load model
    # read name list, todo 这一部分实际是不需要的，只是为了得到 len(label2name) 来载入模型，后续可以修改一下写法
    label2name = {}
    name2label = {}
    with open(args.name_list_file) as fin:
        label = 0
        for line in fin:
            _, name = line.split()
            label2name[label] = name
            name2label[name] = label
            label += 1

    model = ResnetFRModel(n_class=len(label2name), resnet_pretrain=False)
    logging.info(f"Loading checkpoint from {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.to(device)
    model.eval()

    # get embedding
    image1_features = []
    image1_with_flip_features = []
    image2_features = []
    image2_with_flip_features = []
    labels = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(lfw_dataloader)):
            data = {key: data[key].to(device) for key in data}
            batch_image1_features = model(data["image1"], only_feature=True)
            batch_image1_with_flip_features = model(data["image1_with_flip"], only_feature=True)
            batch_image2_features = model(data["image2"], only_feature=True)
            batch_image2_with_flip_features = model(data["image2_with_flip"], only_feature=True)
            image1_features.append(batch_image1_features)
            image1_with_flip_features.append(batch_image1_with_flip_features)
            image2_features.append(batch_image2_features)
            image2_with_flip_features.append(batch_image2_with_flip_features)
            labels.append(data["labels"])

    image1_features = torch.cat(image1_features, 0)
    image1_with_flip_features = torch.cat(image1_with_flip_features, 0)
    image1_features = torch.cat([image1_features, image1_with_flip_features], 1)

    image2_features = torch.cat(image2_features, 0)
    image2_with_flip_features = torch.cat(image2_with_flip_features, 0)
    image2_features = torch.cat([image2_features, image2_with_flip_features], 1)
    labels = torch.cat(labels, 0)

    # calculate accuracy
    dist_f = torch.nn.CosineSimilarity(dim=1)
    thresholds = np.arange(0, 1, 1e-2)
    acc, tpr, fpr = calculate_roc(image1_features, image2_features, labels, dist_f, thresholds)
    print(f"acc: {acc}")

    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    if os.path.exists(args.save_dir):
        raise OSError
    os.mkdir(args.save_dir)
    plt.savefig(os.path.join(args.save_dir, "roc.png"))
    np.save(os.path.join(args.save_dir, "fpr.npy"), fpr)
    np.save(os.path.join(args.save_dir, "tpr.npy"), tpr)
    with open(os.path.join(args.save_dir, "acc.txt"), 'w') as fout:
        fout.write(f"{acc}")


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,
                        default="/home1/sxy/face_recognition/checkpoints/center_loss/with_pretrain/2e-1/lambda8e-2/checkpoint23.pt",
                        help="path to load model checkpoint")
    parser.add_argument("--save_dir", type=str, default="/home1/sxy/face_recognition/results/center_loss_with_pretrain_2e-1_lambda8e-2_23/",
                        help="path to generate roc curve fig")

    parser.add_argument("--image_list_file",
                        default="/home1/sxy/datasets/face_recognition/LFW/cropped_image_list.jsonl",
                        help="each line records the metadata of an image")
    parser.add_argument("--pair_list_file",
                        default="/home1/sxy/datasets/face_recognition/LFW/pairs.txt",
                        help="http://vis-www.cs.umass.edu/lfw/pairs.txt")
    parser.add_argument("--name_list_file", default="/home1/sxy/datasets/face_recognition/CASIA-WebFace/names.txt",
                        help="this file records the name of each identity")

    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help="cpu or cuda")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
