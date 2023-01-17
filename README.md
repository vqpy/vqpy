# VQPy : An object-oriented Video Query Language

[![License](https://img.shields.io/badge/License-Apache_2.0-brightgreen.svg)](https://github.com/uclasystem/VQPy/blob/main/LICENSE)

VQPy is an object-oriented language for modern video analytics. With VQPy, users can express their video queries in a composable and reusable manner with Python. 

VQPy is still under active development. VQPy compiler, which generates a query plan with optimized performance for users' video analytics applications, is working in progress. With VQPy compiler, users can simply focus on the declaration of video queries for their own applications, and multiple optimizations defined in the compiler will be transparently applied to the user’s video analytics pipeline.

## Installation

We have not supported installing VQPy automatically yet. You can follow the steps below to install VQPy manually.

#### Step 1: install VQPy dependences
```
pip3 install lap cython_bbox shapely
```

#### Step 2: install VQPy from source

First, clone vqpy repository from GitHub.
```
git clone https://github.com/uclasystem/VQPy.git
```

Next, add vqpy to your `$PYTHONPATH`. For example, if your local VQPy repository is `$HOME/sources/VQPy`, then you can add your path to VQPy repository to `$PYTHONPATH` with
```
export $PYTHONPATH=$PYTHONPATH:$HOME/sources/VQPy
```
You can test the installation with
```
import vqpy
```

## Getting Started

### Basic usage

In order to declare a video query with VQPy, users need to extend two classes defined in VQPy, namely `Query` and `VObj`. `VObj` defines the objects of interest (e.g., cars, humans, animals, etc.) in one or more video streams, and `Query` defines the video query.  

#### Define a `VObj`

Users can define their own objects of interest, as well as the property in the objects they hope to query on, with a `VObj` class. 

For example,  if we are interested in the vehicle object in the video, and want to query the license plate. We can define a `Vobj` class as below. 

```python
class Vehicle(vqpy.VObjBase):

    @vqpy.property()
    def license_plate(self):
        # infer license plate with vqpy built-in openalpr model
        return self.infer('license_plate', {'license_plate': 'openalpr'})
```

#### Define a `Query`

Users can express their queries through SQL-like constraints with `VObjConstraint`, which is a return value of the `setting` method in their `Query` class. In `VObjConstraint`, users can specify query constraints on the interested object with `filter_cons`, and `select_cons` gives the projection of the properties the query shall return.

The code below demonstrates a query that selects all the `Vehicle` objects whose velocity is greater than 0.1, and chooses the two properties of `track_id`  and `license_plate` for return.

```python
class ListMovingVehicle(vqpy.QueryBase):

    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        filter_cons = {'__class__': lambda x: x == Vehicle,
                       'bbox_velocity': lambda x: x >= 0.1}
        select_cons = {'track_id': None,
                       'license_plate': None}
        return vqpy.VObjConstraint(filter_cons=filter_cons,
                                   select_cons=select_cons)
```

#### Launch the task

Last, we can call `vqpy.launch` to start query video frames.

```python
vqpy.launch(cls_name=vqpy.COCO_CLASSES, # detection class
            cls_type={"car": Vehicle, "truck": Vehicle}, # mappings from detection class to VObj
            tasks=[ListMovingVehicle()], # a list of Queries to apply
            video_path=args.path, # the path of the queried video
            save_folder=args.save_folder # result of query will be saved as a json file in this folder
            )
```

Under the hood, VQPy will automatically select an object detection model that outputs the specified `cls_name`. Multiple video optimizations will be conducted transparently to improve the end-to-end video query performance. 

### Customization

For more details on customization, please refer to the document [here](https://github.com/uclasystem/VQPy/blob/main/docs/frontend.md#customization).

## Examples

We have included two examples for demonstrating VQPy.

- [List red moving vehicle](examples/list_red_moving_vehicle): show license plate of red moving vehicle.
- [Pedestrian Counting](examples/count_person): count the number of pedestrians at the crosswalk.
