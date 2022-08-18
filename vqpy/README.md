## Setup

To setup the current VQPy library, you need to setup a basic pytorch environment. After you have a proper pytorch workspace, the following shell script will help you setup the running environment.

TODO: Add readme for setup openalpr environments.

```shell
mkdir models && cd models
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
mv YOLOX yolo && cd yolo && pip3 install -v -e .
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth
mkdir pretrained && mv yolox_x.pth pretrained && cd ..
git clone https://github.com/xuexingyu24/License_Plate_Detection_Pytorch.git
mv License_Plate_Detection_Pytorch lpdetect && cd ..
pip3 install cython_bbox
```