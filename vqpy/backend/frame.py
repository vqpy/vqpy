from typing import Dict
import numpy
from collections import defaultdict


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
        # Each vobj is a dictionary of properties,
        # (e.g. {"tlbr": [], "score": 0.9}).
        # eg. vobj_data: {"person": [{"tlbr": [], "score": 0.9},],
        self.vobj_data = defaultdict(list)

        # filtered_vobjs is a dictionary of filtered vobjs,
        # where the key is the filtered indexes
        # (each VObjConstraint corresponds to one filter index),
        # and the value is the filtered vobjs
        # (a dictionary of {"class_name": [vobj_indexes]}).
        # eg. filtered_vobjs: {0: {"person": [0, 1]},
        #                      1: {"car": [0, 1, 2], "truck": [0, 1]}}
        self.filtered_vobjs = defaultdict(dict)

    @property
    def video_metadata(self):
        return self._video_metadata

    @property
    def id(self):
        return self._id

    @property
    def image(self):
        return self._image

    def __repr__(self) -> str:
        return f"Frame(id={self.id}, video_metadata={self.video_metadata},\
             vobj_data={self.vobj_data}, filtered_vobjs={self.filtered_vobjs})"
