from typing import Dict, List, Tuple
from ..base.interface import VObjBaseInterface

class SurfaceTrackerBase(object):
    """The surface level tracker base class.
    Objects of this class integrate detections results and associate the
    results with necessary data fields.
    """
    
    input_fields = []       # the required data fields for this tracker

    def update(self, data: List[Dict]) -> Tuple[List[VObjBaseInterface], List[VObjBaseInterface]]:
        raise NotImplementedError
