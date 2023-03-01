"""MultiTracker (a surface tracker) implementation
this tracker separate objects by their classes, and tracks individually
"""

from typing import Callable, Dict, List, Mapping
from vqpy.operator.tracker.base import GroundTrackerBase
from vqpy.query.base import SurfaceTrackerBase

from vqpy.obj.vobj.base import VObjBase, VObjGeneratorType
from vqpy.operator.video_reader import FrameStream
from vqpy.obj.frame import Frame

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

    def update(self, output: List[Dict], last_frame: Frame) -> Frame:
        """Generate the video objects using ground tracker and detection result
        returns: the current tracked/lost VObj instances"""
        detections: Dict[VObjGeneratorType, List[Dict]] = {}

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
                self.tracker_dict[func] = self.tracker(ctx.fps)

        for func, tracker in self.tracker_dict.items():
            # logger.info(f"Multitracking type {func}")
            dets = []
            if func in detections:
                dets = detections[func]
            f_tracked, f_lost = tracker.update(frame_id=ctx.frame_id,
                                               data=dets)
            ctx._objdatas = f_tracked
            for item in f_tracked:
                track_id = item['track_id']
                frame.update_vobjs(func, track_id, item)
            for item in f_lost:
                track_id = item['track_id']
                frame.update_vobjs(func, track_id, None)
            # logger.info(f"tracking done")
        return frame

    def reset(self):
        for tracker in self.tracker_dict.values():
            tracker.reset()
