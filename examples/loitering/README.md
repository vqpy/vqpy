# Detect people loitering example

## Run example

[Video](https://youtu.be/EuLMrUFNRxQ) needs to be downloaded before running the example.

### Running as a script

You can follow the instructions [here](../../README.md#installation) to prepare your environment for VQPy. No other dependency is used.

Then run the script with arguments:

```shell
python VQPy/examples/loitering/main.py
    --path /path/to/video
    --save_folder /path/to/output/folder
    -d /path/to/yolox/model/folder
```

- `--path`: path of video;
- `--save_folder`: the folder to save query result;
- `-d`: directory containing pre-trained models (only model for YOLOX is used).

### Run in Jupyter notebook

Or run [demo.ipynb](./demo.ipynb). The notebook also contains more details about the query.

Notebook is tested in Google Colab, it's advised to use a unused Python3.8 environment if you prefer to run it locally.
