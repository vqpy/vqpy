# Fall detection

To run the example as a script `main.py`, please

- download video from [here](https://youtu.be/ctniCxIdpTY) and specify video location with argument `--path /path/to/video` (as specified in [example readme](../README.md#running-as-a-script))
- download additional model checkpoints, [SPPE FastPose (AlphaPose)](https://drive.google.com/file/d/1IPfCDRwCmQDnQy94nT1V-_NVtTEi4VmU/view?usp=sharing) and [ST-GCN](https://drive.google.com/file/d/1mQQ4JHe58ylKbBqTjuKzpwN2nwKOWJ9u/view?usp=sharing), and place them in the same folder where you put models for object detectors (e.g. pretrained YOLOX model)

    The two model checkpoints should have name `fast_res50_256x192.pth` (SPPE FastPose) and `tsstg-model.pth` (ST-GCN).

    When running the example as the script, specify where the above two models are located with argument `--model`. The command should look like:

    ```shell
    python VQPy/examples/fall_detection/main.py
        --path /path/to/video
        --save_folder /path/to/output/folder
        --model /path/to/pose/detection/model/folder
    ```
