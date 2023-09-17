## Quick Start

### Basic Usage

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

We also support customizing components (e.g. detector) to build the end to end query pipeline. For more details on customization, please refer to the document [here](customization.md).
