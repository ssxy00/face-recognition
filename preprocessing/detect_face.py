# -*- coding: utf-8 -*-
# @Time        : 2021/1/6 16:07
# @Author      : ssxy00
# @File        : detect_face.py
# @Description :

"""
参考：
https://github.com/timesler/facenet-pytorch#guide-to-mtcnn-in-facenet-pytorch

Pipeline: 2
这个脚本会调用 facenet_pytorch 的 MTCNN，为每一张图像生成 landmark
最终生成 json 格式的文件，每一行记录一张图片包含 landmark 的信息 (dict)image_info_with_landmark
image_info_with_landmark = {
'file_path': (str)image_file_path,
'identity_name': (str)identity_name,
'identity_id': (str)identity_id,  # 注意这里 identity_id 是对应到 IMDb 的
'prob': float,
'landmark': List[List[float]],
'box': List[float],
}
"""

import argparse
import jsonlines
import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


class FaceDetector:
    def __init__(self, device, batch_size):
        self.mtcnn = MTCNN(select_largest=False, device=device)
        self.batch_size = batch_size

    def detect_face(self, image_list_file, image_with_landmark_list_file):
        # detect
        frames = []
        landmarks = []
        probs = []
        boxes = []
        image_infos = []
        with open(image_list_file) as fin:
            for image_info in tqdm(jsonlines.Reader(fin)):
                image_infos.append(image_info)
        for image_info in tqdm(image_infos):
            frame = cv2.imread(image_info["file_path"])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)

            # When batch is full, detect faces and reset batch list
            if len(frames) >= self.batch_size:
                batch_boxes, batch_probs, batch_landmarks = self.mtcnn.detect(frames, landmarks=True)
                probs.extend(batch_probs)
                landmarks.extend(batch_landmarks)
                boxes.extend(batch_boxes)
                frames = []

        # save
        with jsonlines.open(image_with_landmark_list_file, 'w') as fout:
            for image_info, prob, box, landmark in zip(image_infos, probs, boxes, landmarks):
                fail_to_detect = landmark is None
                try:
                    fout.write({"file_path": image_info["file_path"],
                                "identity_name": image_info["identity_name"],
                                "identity_id": image_info["identity_id"],
                                "prob": 'none' if fail_to_detect else prob[0].tolist(),
                                "box": 'none' if fail_to_detect else box[0].tolist(),
                                "landmark": 'none' if fail_to_detect else landmark[0].tolist()
                                })
                except:
                    print(type(prob), prob, image_info["file_path"])


def main(args):
    detector = FaceDetector(device=args.device, batch_size=args.batch_size)
    detector.detect_face(image_list_file=args.image_list_file,
                         image_with_landmark_list_file=args.image_with_landmark_list_file)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_list_file",
                        default="/home1/sxy/datasets/face_recognition/CASIA-WebFace/image_list.jsonl",
                        help="input file, each line records the metadata of an image")
    parser.add_argument("--image_with_landmark_list_file",
                        default="/home1/sxy/datasets/face_recognition/CASIA-WebFace/image_with_landmark_list.jsonl",
                        help="output file, each line records the metadata of an image")
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help="cpu or cuda")
    parser.add_argument("--batch_size", default=1024, type=int, help="batch size of the detection process of MTCNN")

    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()
