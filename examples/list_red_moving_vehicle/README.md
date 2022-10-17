# List Red Moving Vehicles example
This example demonstrates how to use VQPy to generate a query that returns the plate numbers of all red moving vehicles.

## Environment preparation
You can follow the instructions [here](../../vqpy/README.md) to prepare your environment for VQPy.

This examples demonstrates how to register a yolox detector to VQPy. So you also need to prepare an enviroment with yolox as well as a pretrained yolox model.
```
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -v -e .

wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth
```

Besides, please use below command to install other dependencies for this example.
```
pip install webcolors ColorDetect opencv-python
```

## Download Dataset
TO BE ADDED.

## Run Example
You can simply use `bash run.sh` to run the example. Below are some arguments in `run.sh` you may want to change.
* `--path`: your own video dataset path.
* `--save_folder`: the folder that you preferred to save the query result.
* `-y`: your pretrained yolox model path.