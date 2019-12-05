# Faster R-CNN

## Requirements


   - Python 3.6 or above
   - Pytorch 1.0
   - CUDA 8.0 or higher

Make sure you meet these before you try to follow commands in the Install section. Note: if you are using `launch-scipy-ml-gpu.sh` you should be fine in terms of meeting these requirements.

## Pretrained VGG and ResNet Models

These models are only useful if you'd like to train a Faster RCNN model. For a pretrained Faster RCNN model see the next section. 

### VGG 16

[Download here](https://www.dropbox.com/s/lsf2g8e398ke6ld/vgg16_caffe.pth?dl=0) and save to the folder `FasterRCNN/data/pretrained_model` as described in the Install section (make sure name is `vgg16_caffe.pth`) (Alternate [link](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth))

### ResNet 50

[Download here](https://www.dropbox.com/s/zpu4sa2fnoxypfv/resnet50_caffe.pth?dl=0) and save to the folder `FasterRCNN/data/pretrained_model` as described in the Install section (make sure name is `resnet50_caffe.pth`)

### Pretrained Faster R-CNN models

Download the trained model from [here](https://www.dropbox.com/s/9kp7gf5tcjurtnn/faster_rcnn_11_7_3723.pth?dl=0) and move it to the directory
`FasterRCNN/models/vgg16/pascal_voc_0712`



## Install

1. Be in FasterRCNN folder and run `mkdir -p data/pretrained_model` and download the models from the links above into this folder (FasterRCNN/data/pretrained_model)

2. Download VOC dataset into the folder named VOCdevkit, then inside the `data` folder run the command `ln -s /path/to/VOCdevkit VOCdevkit2007` (see [link](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) for reference)

3. Go to FasterRCNN folder and run `pip install -r requirements.txt` to install all required python packages.

4. Go to FasterRCNN/lib folder and run `python setup.py build develop`

If this (step 4) fails use `export PYTHONPATH='$PYTHONPATH:/path/to/lib/directory/inside/FasterRCNN/folder'` followed by `python setup.py build develop --install-dir /path/to/lib/directory/inside/FasterRCNN/folder`    

For example:
`export PYTHONPATH='$PYTHONPATH:/datasets/home/home-01/44/344/abkandoi/multi-object-detection/FasterRCNN/lib`    
`python setup.py build develop --install-dir /datasets/home/home-01/44/344/abkandoi/multi-object-detection/FasterRCNN/lib`



## Code organization

    .
    ├── demo.ipynb              # To show how well our model perform on a single image
    ├── train.ipynb             # For a full trainning on the VOC 2007 & 2012 training dataset
    ├── result                  # Folder contains predicted result / training loss / validation loss
    ├── data                    # Folder contains pretrained models
    ├── models                  # Folder stores checkpoints for Faster RCNN models
    ├── lib                     # Source libraries used for loading data, creating Faster RCNN model and evaluating loss
    └── README.md
