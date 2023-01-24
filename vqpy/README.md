## Code Structure

* `obj`: Data type defined in vqpy, including `vobj` and  `frame`, where `vobj` is the objects of interest (e.g., cars, humans, animals, etc.) in video data, and `frame` contains all `vobjs`  on a video frame.
* `operator`: internal operators which generate or operate on the data, including `detector`, `tracker` and `video_reader`.
* `property_lib`: built-in property library calculates the properties for `vobj`.
* `query`: video query interface.
* `class_names`: names of detection classes, like coco classes.
* `utils`: utility functions.
