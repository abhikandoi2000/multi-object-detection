# Faster R-CNN

## Requirements


   - Python 3.6
   - Pytorch 1.0
   - CUDA 8.0 or higher

Make sure you meet these before you try to follow commands in the Install section. Note: if you are using `launch-scipy-ml-gpu.sh` you should be fine in terms of meeting these requirements.

## Install

1. Be in FasterRCNN folder and run `mkdir -p data/pretrained_model` and download the models from the links above into this folder (FasterRCNN/data/pretrained_model)

2. Download VOC dataset into the folder named VOCdevkit, then inside the `data` folder run the command `ln -s /path/to/VOCdevkit VOCdevkit2007` (see [link](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) for reference)

3. Go to FasterRCNN folder and run `pip install -r requirements.txt` to install all required python packages.

4. Go to FasterRCNN/lib folder and run `python setup.py build develop`

If this (step 4) fails use `export PYTHONPATH='$PYTHONPATH:/path/to/lib/directory/inside/FasterRCNN/folder'` followed by `python setup.py build develop --install-dir /path/to/lib/directory/inside/FasterRCNN/folder`    

For example:
`export PYTHONPATH='$PYTHONPATH:/datasets/home/home-01/44/344/abkandoi/multi-object-detection/FasterRCNN/lib`    
`python setup.py build develop --install-dir /datasets/home/home-01/44/344/abkandoi/multi-object-detection/FasterRCNN/lib`
