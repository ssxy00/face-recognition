# -*- coding: utf-8 -*-
# @Time        : 2021/1/22 14:45
# @Author      : ssxy00
# @File        : generate_data_list_for_lfw.py
# @Description :

"""
Pipeline: 1 (test)
这个脚本会读取 LFW 的数据集信息，最终生成 jsonl 格式的文件，每一行记录一张图片的信息 (dict)image_info
image_info = {
'file_path': (str)image_file_path,
'identity_name': (str)identity_name,
'identity_id': (str)identity_id,
}
"""

import os
import jsonlines
import argparse


def generate_identity_dict_for_lfw(image_dir):
    """
    image_dir: images 的根目录
    根目录下有 N 个文件夹，每个文件夹包含一位 identity 的 images，文件夹以 identity name 命名
    关于 lfw 的更多信息可参考 http://vis-www.cs.umass.edu/lfw/README.txt
    :return: {(str)identity_id: {"name": (str)identity_name,
                                 "images": [(str)file_path]
                                 }
              }
    """
    # 读取每个 identity 的图片信息
    identity_dict = {}
    for idx, identity_name in enumerate(os.listdir(image_dir)):
        identity_dict[str(idx)] = {"name": identity_name, "images": []}
        identity_dir = os.path.join(image_dir, identity_name)
        for file_name in os.listdir(identity_dir):
            if file_name[-4:] == ".jpg":
                identity_dict[str(idx)]["images"].append(os.path.join(identity_dir, file_name))
    return identity_dict


def main(args):
    lfw_dict = generate_identity_dict_for_lfw(image_dir=args.image_dir)
    with jsonlines.open(args.image_list_file, 'w') as fout:
        for identity_id in lfw_dict:
            for image_file_path in lfw_dict[identity_id]["images"]:
                fout.write({"file_path": image_file_path,
                            "identity_name": lfw_dict[identity_id]["name"],
                            "identity_id": identity_id,
                            })

def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="/home1/sxy/datasets/face_recognition/LFW/lfw",
                        help="root dir of images")
    parser.add_argument("--image_list_file",
                        default="/home1/sxy/datasets/face_recognition/LFW/image_list.jsonl",
                        help="output file, each line records the metadata of an image")
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()
