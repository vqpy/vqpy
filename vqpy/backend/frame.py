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
