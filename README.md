# face_matching_v1

Babyface detection and matching.

## Requirments
```cmd
# python 3.10.12
conda create -n babyface python=3.10.12
conda activate babyface
# pytorch
pip install "torch==1.13.1+cu117" "torchvision==0.14.1+cu117" --extra-index-url https://download.pytorch.org/whl/cu117
# facenet
pip install facenet-pytorch
```

## Data structure
```
|----iresnet.py
|----network_inf.py
|----main.py
|----magface_epoch_00025.pth
|----class10
|    |----single
|    |    |----b1
|    |    |----b2
|    |    |----...
|    |----group
|    |    |----image1.jpg
|    |    |----...
```
You can download the weight of MagFace from [here](https://drive.google.com/file/d/1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H/view).

## Quick Start
1. Change the path of ```data_folder``` and ```class_name``` in ```main.py```
2. run ```python main.py```
