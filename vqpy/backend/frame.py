from typing import Dict
import numpy


class Frame:
    def __init__(self,
                 video_metadata: Dict,
                 id: int,
                 image: numpy.ndarray):
        self._video_metadata = video_metadata
        self._id = id
        self._image = image
        # vobj_data is a dictionary of detected vobjs of interested class,
        # where the key is the class name and the value is a list of vobjs.
        # Each vobj is a dictionary of properties
        # (e.g. {"class_id": 0, "score": 0.9}).
        self.vobj_data = dict()

    @property
    def video_metadata(self):
        return self._video_metadata

    @property
    def id(self):
        return self._id

    @property
    def image(self):
        return self._image
