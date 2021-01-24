# face-recognition
## References
+ [Center Loss](https://ydwen.github.io/papers/WenECCV16.pdf) Wen, Yandong, et al. "A discriminative feature learning approach for deep face recognition." European conference on computer vision. Springer, Cham, 2016.
+ [CASIA-WebFace](https://arxiv.org/pdf/1411.7923.pdf) Yi, D., et al. "Learning face representation from scratch. arXiv 2014." arXiv preprint arXiv:1411.7923.
+ [MTCNN](https://ieeexplore.ieee.org/abstract/document/7553523) Zhang, Kaipeng, et al. "Joint face detection and alignment using multitask cascaded convolutional networks." IEEE Signal Processing Letters 23.10 (2016): 1499-1503.
+ [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller. Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments. University of Massachusetts, Amherst, Technical Report 07-49, October, 2007.
+ [center-loss.pytorch](https://github.com/louis-she/center-loss.pytorch)
+ [Face Recognition using Tensorflow](https://github.com/davidsandberg/facenet)
+ [Face Recognition Using Pytorch](https://github.com/timesler/facenet-pytorch)
+ [基于MTCNN与insightface的人脸打卡系统](https://blog.nowcoder.net/n/ec11ac368c6c4149834c46e6fe257f81)
+ [sphereface](https://github.com/wy1iu/sphereface/blob/master/preprocess/code/face_align_demo.m)

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
baseline，只用 cross-entropy Loss 训练

#### model A-ResNet
不使用 ImageNet pre-trained 参数
```
# best ckpt: 17
TRAIN_DATA=${TRAIN_LIST_FILE}
VALID_DATA=${VALID_LIST_FILE}
NAME_DATA=${NAME_LIST_FILE}

MODEL_DIR=/path/to/save/model/checkpoints
LOG_DIR=/path/to/save/tensorboard/logs

mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

CUDA_VISIBLE_DEVICES=0 python train.py \
--train_list_file $TRAIN_DATA \
--valid_list_file $VALID_DATA \
--name_list_file $NAME_DATA \
--batch_size 256 \
--n_epochs 100 \
--lr 2e-1 --last_ckpt 0 \
--lr_schedule multi_step --gamma 0.1 \
--save_model_dir $MODEL_DIR \
--save_interval 1 \
--log_dir $LOG_DIR
```


#### model A-ResNet+
使用 ImageNet pre-trained 参数

```
# best ckpt: 16
TRAIN_DATA=${TRAIN_LIST_FILE}
VALID_DATA=${VALID_LIST_FILE}
NAME_DATA=${NAME_LIST_FILE}

MODEL_DIR=/path/to/save/model/checkpoints
LOG_DIR=/path/to/save/tensorboard/logs

mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

CUDA_VISIBLE_DEVICES=0 python train.py \
--train_list_file $TRAIN_DATA \
--valid_list_file $VALID_DATA \
--name_list_file $NAME_DATA \
--resnet_pretrain \
--batch_size 256 \
--n_epochs 50 \
--lr 2e-1 --last_ckpt 0 \
--lr_schedule none \
--save_model_dir $MODEL_DIR \
--save_interval 1 \
--log_dir $LOG_DIR
```

### with center loss
在 softmax 基础上加入 center loss

$\mathcal{L} = \mathcal{L}_S + \mathcal{L}_C$

#### model C-ResNet
不使用 ImageNet pre-trained 参数
```
# best ckpt: 33
TRAIN_DATA=${TRAIN_LIST_FILE}
VALID_DATA=${VALID_LIST_FILE}
NAME_DATA=${NAME_LIST_FILE}

MODEL_DIR=/path/to/save/model/checkpoints
LOG_DIR=/path/to/save/tensorboard/logs

mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

CUDA_VISIBLE_DEVICES=0 python train.py \
--train_list_file $TRAIN_DATA \
--valid_list_file $VALID_DATA \
--name_list_file $NAME_DATA \
--center_loss \
--batch_size 256 \
--n_epochs 100 \
--lr 2e-1 --last_ckpt 0 \
--lr_schedule multi_step --gamma 0.1 \
--save_model_dir $MODEL_DIR \
--save_interval 1 \
--log_dir $LOG_DIR \
--lambda_factor 3e-2
```

#### model C-ResNet+
不使用 ImageNet pre-trained 参数
```
# best ckpt: 16
TRAIN_DATA=${TRAIN_LIST_FILE}
VALID_DATA=${VALID_LIST_FILE}
NAME_DATA=${NAME_LIST_FILE}

MODEL_DIR=/path/to/save/model/checkpoints
LOG_DIR=/path/to/save/tensorboard/logs

mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

CUDA_VISIBLE_DEVICES=0 python train.py \
--train_list_file $TRAIN_DATA \
--valid_list_file $VALID_DATA \
--name_list_file $NAME_DATA \
--resnet_pretrain \
--center_loss \
--batch_size 256 \
--n_epochs 50 \
--lr 2e-1 --last_ckpt 0 \
--lr_schedule none \
--save_model_dir $MODEL_DIR \
--save_interval 1 \
--log_dir $LOG_DIR \
--lambda_factor 3e-3
```

### center initialization: model D
这一部分对 center 的初始化做了改进：
+ 使用 model A-ResNet+ 为训练数据的每个类计算类中心
+ 基于 model A-ResNet+ 再 fine-tune 10 个 epoch

#### calculate center
```
TRAIN_DATA=${TRAIN_LIST_FILE}
NAME_DATA=${NAME_LIST_FILE}
CHECKPOINT_PATH=/path/to/trained/A-ResNet+/checkpoint
CENTER_PATH=/path/to/save/calculated/center

CUDA_VISIBLE_DEVICES=0 python calculate_center.py \
--train_list_file $TRAIN_DATA \
--name_list_file $NAME_DATA \
--checkpoint_path $CHECKPOINT_PATH \
--center_path $CENTER_PATH \
--batch_size 256
```

#### training
```
TRAIN_DATA=${TRAIN_LIST_FILE}
VALID_DATA=${VALID_LIST_FILE}
NAME_DATA=${NAME_LIST_FILE}
CHECKPOINT_PATH=/path/to/trained/A-ResNet+/checkpoint
CENTER_PATH=/path/to/calculated/center

MODEL_DIR=/path/to/save/model/checkpoints
LOG_DIR=/path/to/save/tensorboard/logs

mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

CUDA_VISIBLE_DEVICES=0 python train_center_loss.py \
--train_list_file $TRAIN_DATA \
--valid_list_file $VALID_DATA \
--name_list_file $NAME_DATA \
--center_path $CENTER_PATH \
--checkpoint_path $CHECKPOINT_PATH \
--batch_size 256 \
--n_epochs 10 \
--lr 5e-3 \
--lr_schedule none \
--save_model_dir $MODEL_DIR \
--save_interval 1 \
--log_dir $LOG_DIR 
--lambda_factor 1 \
--ce_factor 0.1 \
```


## Evaluation
### evaluate script
在 LFW 数据集上测试模型，需要用到两个文件：
+ 从 [here](http://vis-www.cs.umass.edu/lfw/lfw.tgz) 下载的图片，经过 preprocessing pipeline 处理后得到的记录每张裁剪后图片的 .jsonl 文件，本地路径为 `IMAGE_LIST_FILE`
+ 从 [here](http://vis-www.cs.umass.edu/lfw/pairs.txt) 下载的 LFW pairs.txt 文件，本地路径为 `PAIR_LIST_FILE`
```
CHECKPOINT_PATH=/path/to/checkpoint/for/evaluation
SAVE_DIR=/path/to/save/evaluate/results

CUDA_VISIBLE_DEVICES=0 python evaluate_on_lfw.py \
--image_list_file $IMAGE_LIST_FILE \
--pair_list_file $PAIR_LIST_FILE \
--name_list_file $NAME_LIST_FILE \
--checkpoint_path $CHECKPOINT_PATH \
--save_dir $SAVE_DIR \
--batch_size 32
```

# results
| Method | Acc. on LFW |
| ---- | ---- |
| model A-ResNet | 92.33 % |
| model A-ResNet + | 93.30 % |
| model C-ResNet | 92.50 % |
| model C-ResNet + | 93.68 % |
| model D | 94.17 % |


