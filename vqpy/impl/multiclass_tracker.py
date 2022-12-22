"""MultiTracker (a surface tracker) implementation
this tracker separate objects by their classes, and tracks individually
"""

from typing import Callable, Dict, List, Mapping, Tuple
from ..base.ground_tracker import GroundTrackerBase
from ..base.surface_tracker import SurfaceTrackerBase

from ..impl.vobj_base import VObjBase, VObjGeneratorType
from ..utils.video import FrameStream
from ..impl.frame import Frame

TrackerGeneratorType = Callable[[FrameStream], GroundTrackerBase]


class MultiTracker(SurfaceTrackerBase):
    """MultiTracker class, separate object by classes and tracks indivdually"""
    input_fields = ["class_id"]

    def __init__(self,
                 tracker: TrackerGeneratorType,
                 cls_name: Mapping[int, str],
                 cls_type: Mapping[str, VObjGeneratorType]):
        """TODO: complete the __init__ docstring"""
        self.tracker = tracker
        self.cls_name = cls_name
        self.cls_type = cls_type
        self.tracker_dict: Dict[VObjGeneratorType, GroundTrackerBase] = {}
        self.vobj_pool: Dict[int, VObjBase] = {}

    def update(self, output: List[Dict], last_frame: Frame
               ) -> Tuple[List[VObjBase], List[VObjBase], Frame]:
        """Generate the video objects using ground tracker and detection result
        returns: the current tracked/lost VObj instances"""
        detections: Dict[VObjGeneratorType, List[Dict]] = {}
        tracked: List[VObjBase] = []
        lost: List[VObjBase] = []

        last_frame_vobjs = last_frame.vobjs
        ctx = last_frame.ctx
        frame = Frame(ctx)
        frame.set_vobjs(last_frame_vobjs)

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
                self.tracker_dict[func] = self.tracker(ctx)

        for func, tracker in self.tracker_dict.items():
            # logger.info(f"Multitracking type {func}")
            dets = []
            if func in detections:
                dets = detections[func]
            f_tracked, f_lost = tracker.update(dets)
            ctx._objdatas = f_tracked
            for item in f_tracked:
                track_id = item['track_id']
                frame.update_vobjs(func, track_id, item)
            tracked = frame.get_tracked_vobjs(func)
            for item in f_lost:
                track_id = item['track_id']
                frame.update_vobjs(func, track_id, None)
            lost = frame.get_lost_vobjs(func)
            # logger.info(f"tracking done")
        return tracked, lost, frame
