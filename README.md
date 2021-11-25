# VRDL_HW2: SVHN object detection
This is homework 2 in NYCU Selected Topics in Visual Recognition using Deep Learning.

## Installation
build the environment via:
```
$ conda env create -f environment.yml
```
And install following packages:
```
$ pip install opencv-python
$ pip install yaml
$ pip install matplotlib
$ pip install tqdm
$ pip install scipy
```
## Prepare Dataset
Split the train data into train_set and validate set
Put the train image in the train folder: ./vrdl_data/images/train_set
Put the validate image in the train folder: ./vrdl_data/images/val
Put the test image in the train folder: ./vrdl_data/images/test
Put the train labels txt in the train folder: ./vrdl_data/labels/train_set
Put the validate labels txt in the train folder: ./vrdl_data/labels/val

Or you can use data_preparation.ipynb to create the image folder:
1. pip install h5py
2. download train.zip and test.zip and unzip to get train/ and test/
3. put train/ into ./vrdl_data/images, and mkdir ./vrdl_data/labels/train
4. put test/ into ./vrdl_data/images
5. Run the data_preparation.ipynb code

The dataset folder should be like:
```
./vrdl_data
  |---train_set.txt
  |---val.txt
  |---test.txt
  |---images
        |---train_set
              |---xxxx.png
              |---xxxx.png
                    .
                    .
        |---val        
              |---xxxx.png
              |---xxxx.png
                    .
                    .
        |---test        
              |---xxxx.png
              |---xxxx.png
                    .
                    .
  |---labels
        |---train_set
              |---xxxx.txt
              |---xxxx.txt
                    .
                    .
        |---val        
              |---xxxx.txt
              |---xxxx.txt
                    .
                    .
```

## Training Code
1. I use the related work **YOLOR** as my object detection model. The official pre-trained (on COCO dataset) weight can be downloaded in https://github.com/WongKinYiu/yolor. In my experiments, **I use the weight of yolor_p6.pt trained on COCO dataset as my pre-trained weights**.
2. Train the model on the given SVHN Dataset. I run our experiments on 1x1080Ti with bacth size of 24 and image size of 640.
```
  $ python train.py --batch-size 16 --img 640 640 --data data.yaml --cfg cfg/yolor_p6.cfg --weights 'yolor_p6.pt' --device 0 --sync-bn --name vrdl_hw_pretrained --hyp hyp.scratch.640.yaml --epochs 300 --worker 4
```

## Evaluation code
Evaluate the model on the test data.
```
  $ python test.py --data data/data.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 2 --cfg cfg/yolor_p6.cfg --weights submission.pt --name submission_result --save-json --task test --names data.names
```

## Download Pretrained Models
Here is the model weight of my final submission. Please download the weights and run the above evaluation code.
+ [Final_submission_weights](https://drive.google.com/file/d/1g-omXmyRrkKfIlSiu8pBUuowMgy9SBms/view?usp=sharing)

## Reference
My howework references the codes in the following repos. Thanks for thier works and sharing.
+ [YOLOR](https://github.com/WongKinYiu/yolor)

