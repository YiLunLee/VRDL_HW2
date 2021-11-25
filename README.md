# VRDL_HW2: SVHN object detection
This is homework 2 in NYCU Selected Topics in Visual Recognition using Deep Learning.

## Installation
build the environment via:
```
$ conda env create -f environment.yml
```

## Prepare Dataset
Split the train data into train_set and validate set
Put the train image in the train folder: ./vrdl_data/images/train_set
Put the validate image in the train folder: ./vrdl_data/images/val
Put the test image in the train folder: ./vrdl_data/images/test
Put the train labels txt in the train folder: ./vrdl_data/labels/train_set
Put the validate labels txt in the train folder: ./vrdl_data/labels/val
(You can refer to data_preparation.ipynb for details.)
The dataset folder should be like:
```
./vrdl_data
  |---train_set.txt
  |---val.txt
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
  $ 
```

## Evaluation code
Evaluate the model on the test data.
```
  $ 
```

## Download Pretrained Models
Here is the model weight of my final submission. Please download the weights and run the above evaluation code.
+ [Final_submission_weights]()

## Reference
My howework references the codes in the following repos. Thanks for thier works and sharing.
+ [YOLOR](https://github.com/WongKinYiu/yolor)

