# -*- coding: utf-8 -*-
# @Time        : 2021/1/6 16:08
# @Author      : ssxy00
# @File        : align_face.py
# @Description :

"""
参考：
https://github.com/wy1iu/sphereface/blob/master/preprocess/code/face_align_demo.m
https://blog.nowcoder.net/n/ec11ac368c6c4149834c46e6fe257f81

Pipeline: 3
这个脚本会根据图像标注的 landmark 做 similarity transformation，并 crop 到 112 * 96 的尺寸
最终生成 jsonl 格式的文件，每一行记录一张图片的信息（包含 align 后的图片路径） (dict)cropped_image_info
cropped_image_info = {
'file_path': (str)image_file_path,
'cropped_file_path': (str)cropped_image_file_path,
'identity_name': (str)identity_name,
'identity_id': (str)identity_id,  # 注意这里 identity_id 是对应到 IMDb 的
'prob': float,
'landmark': List[List[float]],
'box': List[float],
}
"""

import os
import argparse
import jsonlines
import cv2
import numpy as np
from skimage import transform as trans

IMAGE_SIZE = (112, 96)
COORD5POINT = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]], dtype=np.float32)


def align(frame, landmark):
    # skimage affine
    tform = trans.SimilarityTransform()
    tform.estimate(landmark, COORD5POINT)
    M = tform.params[0:2, :]
    transformed_frame = cv2.warpAffine(frame, M, (IMAGE_SIZE[1], IMAGE_SIZE[0]), borderValue=0.0)
    return transformed_frame


def main(args):
    count = 0
    image_idx = 1
    cropped_image_dir = "/home1/sxy/datasets/face_recognition/CASIA-WebFace/cropped_images"
    with open(args.image_with_landmark_list_file) as fin, jsonlines.open(args.cropped_image_list_file, 'w') as fout:
        for image_info in jsonlines.Reader(fin):
            frame = cv2.imread(image_info["file_path"])
            if image_info["landmark"] == 'none':
                count += 1
                continue
            cropped_frame = align(frame, np.array(image_info["landmark"], dtype=np.float32))

            cropped_file_path = os.path.join(cropped_image_dir, f"{image_idx}.jpg")
            cv2.imwrite(cropped_file_path, cropped_frame)
            cropped_image_info = {"cropped_file_path": cropped_file_path}
            cropped_image_info.update(image_info)
            fout.write(cropped_image_info)
            image_idx += 1
            if image_idx % 10000 == 0:
                print(f"already processing {image_idx} images")
    print(count)  # 2471


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_with_landmark_list_file",
                        default="/home1/sxy/datasets/face_recognition/CASIA-WebFace/image_with_landmark_list.jsonl",
                        help="input file, each line records the metadata of an image")
    parser.add_argument("--cropped_image_list_file",
                        default="/home1/sxy/datasets/face_recognition/CASIA-WebFace/cropped_image_list.jsonl",
                        help="output file, each line records the metadata of an image")
    parser.add_argument("--cropped_image_dir",
                        default="/home1/sxy/datasets/face_recognition/CASIA-WebFace/cropped_images",
                        help="output dir of cropped images")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
