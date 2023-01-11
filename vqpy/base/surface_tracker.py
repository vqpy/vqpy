"""the surface-level tracker base class"""

from typing import Dict, List

from ..base.interface import FrameInterface


class SurfaceTrackerBase(object):
    """The surface level tracker base class.
    Objects of this class integrate detections results and associate the
    results with necessary data fields.
    """

    input_fields = []       # the required data fields for this tracker

    def update(self, data: List[Dict]) -> FrameInterface:
        """Generate the video objects using ground tracker and detection result
        returns: the current tracked/lost VObj instances"""
        raise NotImplementedError
