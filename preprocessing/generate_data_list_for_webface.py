# -*- coding: utf-8 -*-
# @Time        : 2021/1/6 16:00
# @Author      : ssxy00
# @File        : generate_data_list_for_webface.py
# @Description :

"""
Pipeline: 1
这个脚本会读取 CASIA-WebFace 的数据集信息，最终生成 jsonl 格式的文件，每一行记录一张图片的信息 (dict)image_info
image_info = {
'file_path': (str)image_file_path,
'identity_name': (str)identity_name,
'identity_id': (str)identity_id,  # 注意这里 identity_id 是对应到 IMDb 的
}
"""

import os
import jsonlines
import argparse


def generate_identity_dict_for_webface(image_dir, name_list_file):
    """
    CASIA-WebFace 的相关文件有两个部分：
    image_dir: images 的根目录
    name_list_file: 每一个 identity 的名字，从 https://groups.google.com/g/cmu-openface/c/Xue_D4_mxDQ 下载
    :return: {(str)identity_id: {"name": (str)identity_name,
                                 "images": [(str)file_path]
                                 }
              }
    """
    # 读取每个 identity 的名字
    identity_out_of_lfw = {}
    with open(name_list_file) as fin:
        for line in fin:
            identity_id, name = line.split()
            identity_out_of_lfw[identity_id] = {'name': name, 'images': []}
    # 读取每个 identity 的图片信息
    for identity_id in identity_out_of_lfw:
        identity_dir = os.path.join(image_dir, identity_id)
        for file_name in os.listdir(identity_dir):
            if file_name[-4:] == ".jpg":
                identity_out_of_lfw[identity_id]['images'].append(os.path.join(identity_dir, file_name))
    return identity_out_of_lfw


def main(args):
    webface_dict = generate_identity_dict_for_webface(image_dir=args.image_dir, name_list_file=args.name_list_file)
    with jsonlines.open(args.image_list_file, 'w') as fout:
        for identity_id in webface_dict:
            for image_file_path in webface_dict[identity_id]["images"]:
                fout.write({"file_path": image_file_path,
                            "identity_name": webface_dict[identity_id]["name"],
                            "identity_id": identity_id,
                            })

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="/home1/sxy/datasets/face_recognition/CASIA-WebFace/CASIA-WebFace",
                        help="root dir of images")
    parser.add_argument("--name_list_file", default="/home1/sxy/datasets/face_recognition/CASIA-WebFace/names.txt",
                        help="this file records the name of each identity")
    parser.add_argument("--image_list_file",
                        default="/home1/sxy/datasets/face_recognition/CASIA-WebFace-new/image_list.jsonl",
                        help="output file, each line records the metadata of an image")
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()
