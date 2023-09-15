# VQPy : An Object-Oriented Approach to Modern Video Analytics

[![License](https://img.shields.io/badge/License-Apache_2.0-brightgreen.svg)](https://github.com/uclasystem/VQPy/blob/main/LICENSE)

VQPy is a Python-based video analytics library, designed to address two major issues in today's video analytics. 

- The logic of typical video queries focuses on video objects (e.g., human and cars) and their interactions; it is often awkward to express such logic with a SQL-like language that builds on structured data, a data model fundamentally different from objects and relations.
- A video pipeline often consists of multiple fragments each involving a different vision algorithm or NN. It is challenging to connect these fragments and orchestrate them in a natural flow with little human effort.

Building on the insight of object orientation, VQPy solves these problems by presenting a <b> video-object-oriented</b> view to analytics developers.  VQPy allows a complex query to be expressed with a very small number of lines of code. VQPy supports query sharing and composition---finding a red car can build on an existing query that finds a general car; monitoring traffic for a city can build on car monitoring queries built for individual districts and intersections, thereby significantly simplifying development and deployment. VQPy allows different objects and relations to be registered with different trackers and detectors,  connecting different fragments of a pipeline naturally with object-oriented constructs such as inheritance and encapsulation. 

Please check out our examples below for details. 

The development of VQPy was initiated by [Harry Xu](http://www.cs.ucla.edu/~harryxu)'s group at UCLA, and has evovled over the time into a community effort, involving folks from both academia and industry. VQPy is now part of Cisco's [DeepVision platform](https://research.cisco.com/research-projects/deep-vision) which is deployed world-wide to support complex queries over customer videos.

## Installation

<details><summary>Show installation details</summary>
<p>

### Conda

VQPy is developed and tested on Linux. Therefore we highly recomand you to try VQPy on Linux, to avoid encountering some unknown errors.

You can follow the steps below to install VQPy.

#### Step 0: install conda
We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the Python environment as follows:
```shell
conda create -n vqpy python=3.8  # "vqpy" is conda environment name, you can use any name you like.
conda activate vqpy
```

#### Step 1: install VQPy
We haven't publish vqpy to pypi yet. You can use the commands below to install VQPy from Github.
```shell
pip install torch torchvision numpy==1.23.5 cython
pip install 'vqpy @ git+https://github.com/vqpy/vqpy.git'
```

#### Step 2: test installation
You can test whether vqpy has been successfully installed with
```python
import vqpy
from vqpy import query
```

### Docker

You can also try run vqpy in a Docker container by following the below steps.

#### Step 0: prerequisites
You need to have docker installed on your system (you can download Docker [here](https://www.docker.com/get-started/))

You also need to clone the VQPy code repository into your local (since we need to build a docker image).

#### Step 1: Build the Docker image
Navigate to the directory containing the Dockerfile and use the docker build command to build your Docker image. Replace vqpy_image with the name you want to give to your Docker image.
```shell
docker build -t vqpy_image .
```

#### Step 2: Run the docker container
After building the image, you can run the container with the docker run command. This command creates a new Docker container from the Docker image and starts it. Replace vqpy_container with the name you want to give to your Docker container.
```shell
docker run -it --name vqpy_container vqpy_image /bin/bash
```
</p>
</details>

## Overview
Below is an architecture graph which describes the VQPy framework.
<p align="center">
  <img src="docs/resources/Architecture.png" width="800" title="architecture">
</p>

In the frontend, we provide interfaces for users to declare their query, i.e `VObj` and `Query`.

In the backend, vqpy automatically select the best plan to execute the query, within user specified budget (e.g. accuracy, inference time, .etc).

VQPy also provides a library containing rich models and property functions that help save users' efforts to build queries.

Note that users can also register their own models and functions to vqpy library, to enable reusability between different queries and customize their own end to end video query pipeline.

## Quick Start

### Basic usage

In order to declare a video query with VQPy, users need to extend two classes defined in VQPy, namely `Query` and `VObj`. `VObj` defines the objects of interest (e.g., cars, humans, animals, etc.) in one or more video streams, and `Query` defines the video query.  

#### Step 1. Define a `VObj`

Users can define their own objects of interest, as well as the property in the objects they hope to query on, with a `VObj` class. 

To define a `VObj` class, users are required to inherit the `vqpy.VObjBase` class. User can also define properties in the `VObj` that can be used in query (like "license plate" of a "Vehicle"). To define a property, the definition should start either with a `@vqpy.property` decorator or a `@vqpy.cross_object_property` decorator, where `@vqpy.property` indicates that the calculation of the property only based on the attributes of **the VObj instance**, and `@vqpy.cross_vobj_property` indicates that the calculation of the property requires the arributes of **other Vobj instances** on the same frame.

For example,  if we are interested in the vehicle object in the video, and want to query the license plate. We can define a `Vehicle` class as below.

```python
class Vehicle(vqpy.VObjBase):

    @vqpy.property()
    def license_plate(self):
        # infer license plate with vqpy built-in openalpr model
        return self.infer('license_plate', {'license_plate': 'openalpr'})
```

And if we want to query the owner of a baggage object where the baggage's owner is a person object who is closest to the baggage. We can define our interested `VObj`s as below. Note that the `owner` property of `Baggage` `VObj` should be decorcated with `@vqpy.cross_vobj_property`.

```python
class Person(vqpy.VObjBase):
    pass

class Baggage(vqpy.VObjBase):

    @vqpy.cross_vobj_property(vobj_type=Person, vobj_num="ALL", vobj_input_fields=("track_id", "tlbr"))
    def owner(self, person_id_tlbrs):
        pass
```

To retrieve the attribute values in property calculation, you can use `self.getv("property_name", index={history_index})` or `self.infer`. For more information about how to retrieve attribute values in VObj, you can refer to document [here](docs/frontend.md#retrieve-property-value-in-vobj).

You can find more details about `VObj` as well as the decorators in our [VObj API document](vqpy/obj/vobj/base.py) and [VObj decorator API document](vqpy/obj/vobj/wrappers.py) (Currently is the docstring in our source code).


#### Step 2. Define a `Query`

Users can express their queries through SQL-like constraints with `VObjConstraint`, which is a return value of the `setting` method in their `Query` class. In `VObjConstraint`, users can specify query constraints on the interested `VObj` with `filter_cons`, and `select_cons` gives the projection of the properties the query shall return.

Note that the keys for both `filter_cons` and `select_cons` should be strings of property names, where the property name can either be a vqpy built-in property name or a user declared property name of the interested `VObj`. The value of `filter_cons` dictionary should be a boolean function, the `VObj` instances which satisfy all the boolean functions in `filter_cons` will be selected as query results. The value of `select_cons` dictionary is the postprocessing function applied on the property for query output. If you want to directly output the property value without any postprocessing functions, you can just set the value to `None.`

The code below demonstrates a query that selects all the `Vehicle` objects whose velocity is greater than 0.1, and chooses the two properties of `track_id`  and `license_plate` as outputs.

```python
class ListMovingVehicle(vqpy.QueryBase):

    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        filter_cons = {'__class__': lambda x: x == Vehicle,
                       'velocity': lambda x: x >= 0.1}
        select_cons = {'track_id': None,
                       'license_plate': None}
        return vqpy.VObjConstraint(filter_cons=filter_cons,
                                   select_cons=select_cons)
```

We also provide boolean functions that can be used in query, like `vqpy.query.continuing`, which checks whether a condition function continues to be true for a certain duration. For the detailed usage, please refer to our [People Loitering](examples/loitering) example.

#### Step 3. Launch the task

After declaring `VObj` and `Query`, users should call the `vqpy.launch` function to deploy the video query, below is an example of this interface. Users should specify the detection class they expect, and map the object detection result to the defined VObj types.

```python
vqpy.launch(cls_name=vqpy.COCO_CLASSES, # detection class
            cls_type={"car": Vehicle, "truck": Vehicle}, # mappings from detection class to VObj
            tasks=[ListMovingVehicle()], # a list of Queries to apply
            video_path=args.path, # the path of the queried video
            save_folder=args.save_folder # result of query will be saved as a json file in this folder
            )
```

Under the hood, VQPy will automatically select an object detection model that outputs the specified `cls_name`, and decide the best predicates order to execute the query. Multiple video optimizations will be conducted transparently to improve the end-to-end video query performance.

### Advanced Usage

We also support customizing components (e.g. detector) to build the end to end query pipeline. For more details on customization, please refer to the document [here](docs/customization.md).

## Examples

We have included several real-world video analytics examples for demonstrating VQPy.

- [People Loitering](examples/loitering): Detects and sends alerts when individuals loiter in designated areas beyond set time thresholds. ([DeepVision Demo](examples/loitering/loitering-vqpy-DV-demo.mov))
- [Queue Analysis](examples/queue_analysis/): Analyze queue metrics such as the number of people waiting, average/min/max waiting times, etc. ([DeepVision Demo](examples/queue_analysis/vqpy-DeepVision.mov))
- [Fall Detection](examples/fall_detection): Recognize fallen people in a video.
- [List red moving vehicle](examples/list_red_moving_vehicle): show the license plates of red moving vehicles.
- [People Counting](examples/people_counting): count the number of people heading different directions.
- [Unattended Baggage Detection](examples/unattended_baggage): detect unattended baggages.

## Getting Support

- Use VQPy's [slack channel](https://join.slack.com/t/vqpy/shared_invite/zt-1mnq3uh9v-o2~uNUnRQRudNTrYCNHeUA) to ask questions and share ideas!
- Create a github [issue](https://github.com/vqpy/vqpy/issues).

## Acknowledgements

We are grateful to the generous support from:

<img src="https://upload.wikimedia.org/wikipedia/commons/1/12/NSF.svg" height=100in weight=100in />| <img src="https://1000logos.net/wp-content/uploads/2016/11/Cisco-logo.png" height=100in weight=200in />

