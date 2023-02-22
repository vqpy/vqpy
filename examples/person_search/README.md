# Person search
This example demonstrates how to use VQPy to search the target person from videos across different camera views.

## Environment Preparation
You can follow the instructions [here](../../README.md#installation) to prepare your environment for VQPy.

Besides, the additional models and pretrained weights for person search (Fast-ReID) are loaded by `torch.hub`, please install other dependencies for additional models as following the instructions [here](https://github.com/JDAI-CV/fast-reid/blob/master/INSTALL.md) for this example.

```
pip install -r https://raw.githubusercontent.com/JDAI-CV/fast-reid/master/docs/requirements.txt
```

## Download Dataset

Download example video from [here](https://drive.google.com/file/d/1xkCr6uY-wp0ZdhJfEkhd_7XOc0qNmOcq/view?usp=sharing) and specify video location with argument `--path /path/to/video` (as specified in [example readme](../README.md#running-as-a-script))


## Run Example
You can simply use `python main.py` to run the example. Below are the arguments you need to specify.
* `--path`: your own video dataset path.
* `--query_folder`: the folder containing query images.
* `--save_folder`: the folder that you preferred to save the query result.
