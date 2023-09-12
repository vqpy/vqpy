# VQPy Examples

## Basic Examples

- [Fall Detection](examples/fall_detection): Recognize fallen people in a video. [video](https://youtu.be/ctniCxIdpTY)
- [List red moving vehicle](examples/list_red_moving_vehicle): show the license plates of red moving vehicles.
- [People Loitering](examples/loitering): detect suspicious activities of person loitering around. [video](https://youtu.be/EuLMrUFNRxQ)
- [Person Search](examples/person_search): retrieve the target person in videos.
- [People Counting](examples/people_counting): count the number of people heading different directions.
- [Unattended Baggage Detection](examples/unattended_baggage): detect unattended baggages. [video](https://www.kaggle.com/datasets/szahid405/baggage?select=baggage.mp4)
- [Queue Analysis](examples/queue_analysis): analysis the max/min/average waiting time in a queue. [video](https://www.youtube.com/watch?v=KMJS66jBtVQ)

## Run Examples

### Run python script

You can follow the instructions [here](../README.md#installation) to prepare your environment for VQPy.

You can download videos with the links in the above example list to prepare data for the examples. Note that the Fall Detection example requires downloading pre-trained models; to run the Fall Detection example, you can follow the instructions [here](fall_detection/README.md) instead.

Then run the script with arguments:

```shell
python VQPy/examples/example/script.py
    --path /path/to/video
    --save_folder /path/to/output/folder
```

- `--path`: path of video;
- `--save_folder`: the folder to save query result;

### Run in Jupyter notebook

You can easily try our examples with Jupyter notebook (The `demo.ipynb` in each example's directory). The notebooks will guide you through running video queries with VQPy from scratch.

You can directly open and run the notebooks on Google Colab. However, if you prefer to run the notebooks locally, we recommend creating a new environment for VQPy with python 3.8 installed.
