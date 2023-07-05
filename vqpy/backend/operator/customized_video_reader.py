from vqpy.backend.operator.base import Operator
from vqpy.backend.frame import Frame
from abc import abstractmethod
from typing import Dict


class CustomizedVideoReader(Operator):
    def __init__(self):
        metadata = self.get_metadata()
        assert isinstance(metadata, dict), "get_metadata must return a dictionary."
        assert "fps" in metadata, "The return value of get_metadata must have \
        a key 'fps'."
        self.metadata = metadata

    @abstractmethod
    def get_metadata(self) -> Dict:
        raise NotImplementedError("Please implement the method get_metadata.")

    @abstractmethod
    def _next(self) -> Dict:
        raise NotImplementedError("Please implement the method _next.")

    @abstractmethod
    def has_next(self) -> bool:
        raise NotImplementedError("Please implement the method has_next.")

    def next(self) -> Frame:
        result = self._next()

        # validate the result
        assert isinstance(result, dict), "The method must return a dictionary."
        assert "frame_id" in result, "The dictionary must have a key 'frame_id'."
        assert "image" in result, "The dictionary must have a key 'image'."

        return Frame(video_metadata=self.metadata,
                     id=result["frame_id"],
                     image=result["image"])
