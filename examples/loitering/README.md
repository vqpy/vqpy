## Loitering Detection

The example demonstrates how to use VQPy to detect and send alerts when individuals loiter in specified areas beyond set time thresholds.
Real-time loitering detection is crucial for smart city safety, preventing potential risks such as suicides on infrastructure, retail store burglaries, and parking lot assaults.

### Environment preparation
You can follow the instructions [here](../../README.md#installation) to prepare your environment for VQPy.

### Run Example
You can use `python new_frontend.py` to run the example. Below are the arguments you need to specify.
* `--path`: your own video dataset path.
* `--save_folder`: the folder that you preferred to save the query result.
* `--polygon`: the polygon coordinates of the loitering area.
* `--time_warning`: the time threshold for warning.
* `--time_alert`: the time threshold for alert.

### Deep Vision Demo
![Loitering Detection](./demo.assets/loitering-vqpy-DV-demo.gif)