# face-recognition
## Install
+ python==3.6.8
+ pytorch==1.4.0
```
conda create -n cv python==3.6.8
conda activate cv
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
python -m pip install -r requirements.tx
```

## Preprocessing: Pipeline

### Stage 1: extract meta data 
#### CASIA-WebFace
读取 CASIA-WebFace 的图片信息

这个部分的会用到两个文件（夹）：
+ 从 [Images](http://shujujishi.com/dataset/9907a487-eb1d-438c-8bbf-52700918ef98.html)下载的 CASIA-WebFace 的图片集，本地路径是 `IMAGE_DIR`
+ 从 [names](https://groups.google.com/g/cmu-openface/c/Xue_D4_mxDQ) 下载的 CASIA-WebFace Identity name 与 id 的对应，本地路径是 `NAME_LIST_FILE`
并且会生成一个 .jsonl 的文件，每一行是一张图片的信息，本地路径是 `IMAGE_LIST_FILE`

```
python generate_data_list_for_webface.py \
--image_dir ${IMAGE_DIR} \
--name_list_file ${NAME_LIST_FILE} \
--image_list_file ${IMAGE_LIST_FILE}
```

#### LFW
读取 LFW 的图片信息

这个部分需要用到从 [here](http://vis-www.cs.umass.edu/lfw/lfw.tgz) 下载的数据，解压后的路径为 `IMAGE_DIR`，
读取的图片信息保存在 `IMAGE_LIST_FILE`

```
python generate_data_list_for_lfw.py \
--image_dir ${IMAGE_DIR} \
--image_list_file ${IMAGE_LIST_FILE}
```

### Stage 2: detect
调用 [MTCNN](https://arxiv.org/pdf/1604.02878.pdf) 识别每个图像五个点的 landmark，这里使用的是 [facenet-pytorch](https://github.com/timesler/facenet-pytorch#guide-to-mtcnn-in-facenet-pytorch)

这一部分会读取 `IMAGE_LIST_FILE` 中每一行的图片信息，并添加 landmark 后保存到 `IMAGE_WITH_LANDMARK_LIST_FILE`
```
python detect_face.py \
--image_list_file ${IMAGE_LIST_FILE} \
--image_with_landmark_list_file ${IMAGE_WITH_LANDMARK_LIST_FILE}
```

### Stage 3: align
使用 Similarity Transformation 对图像做变换，并切割到 112 * 96 的尺寸
这一部分会读取 `IMAGE_WITH_LANDMARK_LIST_FILE` 中每一行的图片信息，并添加变换后的图片路径后保存到 `CROPPED_IMAGE_LIST_FILE`，其中变换后的图片会保存在 `CROPPED_IMAGE_DIR` 目录下
```
python align_face.py \
--image_with_landmark_list_file ${IMAGE_WITH_LANDMARK_LIST_FILE} \
--cropped_image_list_file ${CROPPED_IMAGE_LIST_FILE} \
--cropped_image_dir ${CROPPED_IMAGE_DIR}
```

### Stage 4: split
将 CASIA-WebFace 数据集按照 9:1 的比例划分为训练集 `TRAIN_LIST_FILE` 和验证集 `VALID_LIST_FILE`

| training set | validation set | total |
| ---- | ---- | ---- |
| 442,494 | 49,166 | 491,660 |

```
python random_split.py \
--cropped_image_list_file ${CROPPED_IMAGE_LIST_FILE} \
--train_list_file ${TRAIN_LIST_FILE} \
--valid_list_file ${VALID_LIST_FILE}
```

## Training
### without center-loss
目前在训练一个 baseline，只用 cross-entropy Loss 训练，先不加 center loss，作为参照

| arguments |  |
| ---- | ---- |
| optimizer | SGD |
| batch size | 256 |
| n_epochs | ? |
| lr | ? (根据 valid loss 手动调) |

### TODO
- [ ] center loss
- [ ] test on LFW


## Evaluation
### preprocessing

