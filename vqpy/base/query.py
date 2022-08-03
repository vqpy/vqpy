from typing import List

from vqpy.utils.video import FrameStream
from vqpy.impl.vobj_base import VObjBase

class QueryBase:
    def attach(self, ctx: FrameStream):
        """attach the working stream with the query object, will be called at the beginning of tracking"""
        self._ctx = ctx
    
    def apply(self, tracks: List[VObjBase]) -> List[VObjBase]:
        """
        Apply something required to the per-frame updated tracks.
        tracks: the list of all VQPy objects appeared in this frame.
        """
        pass