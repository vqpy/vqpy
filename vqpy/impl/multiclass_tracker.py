from typing import Callable, Dict, List, Mapping, Tuple
from ..base.ground_tracker import GroundTrackerBase
from ..base.surface_tracker import SurfaceTrackerBase

from ..impl.vobj_base import VObjBase, VObjGeneratorType
from ..utils.video import FrameStream

TrackerGeneratorType = Callable[[FrameStream], GroundTrackerBase]

class MultiTracker(SurfaceTrackerBase):
    input_fields = ["class_id"]
    
    def __init__(self,
                 tracker: TrackerGeneratorType,
                 ctx: FrameStream,
                 cls_name: Mapping[int, str],
                 cls_type: Mapping[str, VObjGeneratorType]):
        self.ctx = ctx
        self.tracker = tracker
        self.cls_name = cls_name
        self.cls_type = cls_type
        self.tracker_dict: Dict[VObjGeneratorType, GroundTrackerBase] = {}
        self.vobj_pool: Dict[int, VObjBase] = {}
    
    def update(self, output: List[Dict]) -> Tuple[List[VObjBase], List[VObjBase]]:
        detections: Dict[VObjGeneratorType, List[Dict]] = {}
        tracked: List[VObjBase] = []
        lost: List[VObjBase] = []
        
        for obj in output:
            name = self.cls_name[obj['class_id']]
            if name not in self.cls_type:
                continue
            func = self.cls_type[name]
            if func not in detections:
                detections[func] = []
            detections[func].append(obj)
    
        for func, dets in detections.items():
            if func not in self.tracker_dict:
                self.tracker_dict[func] = self.tracker(self.ctx)
    
        for func, tracker in self.tracker_dict.items():
            # logger.info(f"Multitracking type {func}")
            dets = []
            if func in detections:
                dets = detections[func]
            f_tracked, f_lost = tracker.update(dets)
            self.ctx._objdatas = f_tracked
            for item in f_tracked:
                track_id = item['track_id']
                if track_id not in self.vobj_pool:
                    self.vobj_pool[track_id] = func(self.ctx)
                self.vobj_pool[track_id].update(item)
                tracked.append(self.vobj_pool[track_id])
            for item in f_lost:
                track_id = item['track_id']
                self.vobj_pool[track_id].update(None)
                lost.append(self.vobj_pool[track_id])
            # logger.info(f"tracking done")
        
        return tracked, lost

