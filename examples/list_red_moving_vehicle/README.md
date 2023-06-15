# List Red Moving Vehicles example
This example demonstrates how to use VQPy to generate a query that returns the plate numbers of all red moving vehicles.

## Environment preparation
You can follow the instructions [here](../../README.md#installation) to prepare your environment for VQPy.

Besides, please use below command to install other dependencies for this example.
```
pip install webcolors ColorDetect opencv-python
```

And you can follow the instructions [here](https://github.com/openalpr/openalpr#binaries) to install openalpr.

## Download Dataset
You can use your own video data as needed. The sample data ([license-10s.mp4](license-10s.mp4)) is a snippet from [Kaggle Automatic Number Plate Recognition Dataset](https://www.kaggle.com/datasets/aslanahmedov/number-plate-detection?resource=download&select=TEST)

## Run Example
You can simply use `python main.py` to run the example. Below are the arguments you need to specify.
* `--path`: your own video dataset path.
* `--save_folder`: the folder that you preferred to save the query result.

If you want to try the old frontend, you can use `python old_frontend.py` with the same arguments.