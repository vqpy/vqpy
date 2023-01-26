# VQPy Examples

## Basic Examples

- [List red moving vehicle](list_red_moving_vehicle): show license plate of red moving vehicle.
- [Pedestrian Counting](count_person): count the number of pedestrians at the crosswalk.
- [Detect people loitering](loitering): detect people loitering at specified region. [video](https://youtu.be/EuLMrUFNRxQ)
- [Unattended baggage](unattended_baggage): detect baggage with people around. [video](https://www.kaggle.com/datasets/szahid405/baggage?select=baggage.mp4)
- [Fall detection](fall_detection): detect people's pose. [video](https://youtu.be/ctniCxIdpTY); extra pretrained models required to run the query, see [here](fall_detection/README.md) for instructions

## Running examples

Videos from the links above needs to be downloaded before running examples. Some examples (e.g. fall detection) requires additional pretrained models, see README's in their corresponding folders for instructions.

### Running as a script

You can follow the instructions [here](../../README.md#installation) to prepare your environment for VQPy. No other dependency is used.

Then run the script with arguments:

```shell
python VQPy/examples/example/script.py
    --path /path/to/video
    --save_folder /path/to/output/folder
    -d /path/to/yolox/model/folder
```

- `--path`: path of video;
- `--save_folder`: the folder to save query result;
- `-d`: directory containing pre-trained models (only model for YOLOX is used unless explicitly specified).

### Run in Jupyter notebook

Or follow `demo.ipynb` in each example's directory. The notebooks contain more details about the queries.

Notebook is tested in Google Colab, it's advised to use a unused Python3.8 environment if you prefer to run it locally.
