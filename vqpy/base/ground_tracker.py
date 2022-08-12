from typing import Dict, List, Tuple
from ..utils.video import FrameStream

class GroundTrackerBase(object):
    """The ground level tracker base class.
    Objects of this class approve detections results and associate the
    results with necessary data fields.
    """
    
    input_fields = []       # the required data fields for this tracker
    output_fields = []      # the necessary data fields this tracker can generate
    
    def __init__(self, stream: FrameStream):
        raise NotImplementedError

    def update(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        raise NotImplementedError
