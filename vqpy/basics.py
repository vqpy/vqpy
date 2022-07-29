import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import cv2
import torch
from loguru import logger

from vqpy.utils import *
from vqpy.video_loader import FrameStream
from vqpy.objects import VObjBase, VObjGeneratorType

class PredictorBase(object):
    def inference(self, img: np.ndarray) -> List[Dict]:
        raise NotImplementedError

class TrackerBase(object):
    # tracker filters the vobjects and assign TRACKIDs to supported tracks
    # hence tracker only take object values as inputs, and return the filter result
    def __init__(self, stream: FrameStream):
        raise NotImplementedError

    def update(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        raise NotImplementedError

# TODO: add tracker base to enable different trackers

class QueryBase:
    def attach(self, ctx: FrameStream):
        """attach the working stream with the query object, will be called at the beginning of tracking"""
        self._ctx = ctx
    
    def apply(self, tracks: List[VObjBase]) -> None:
        """
        Apply something required to the per-frame updated tracks.
        tracks: the list of all VQPy objects appeared in this frame.
        """
        pass
    
    def finalize() -> None:
        """This function will be called after the video analysis is completed"""
        pass

TrackerGeneratorType = Callable[[FrameStream], TrackerBase]

class MultiTracker(object):
    input_fields = ["class_id"]
    
    def __init__(self,
                 tracker: TrackerGeneratorType,
                 stream: FrameStream,
                 cls_name: Mapping[int, str],
                 cls_type: Mapping[str, VObjGeneratorType]):
        self.stream = stream
        self.tracker = tracker
        self.cls_name = cls_name
        self.cls_type = cls_type
        self.tracker_dict: Dict[VObjGeneratorType, TrackerBase] = {}
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
                self.tracker_dict[func] = self.tracker(self.stream)
    
        for func, tracker in self.tracker_dict.items():
            # logger.info(f"Multitracking type {func}")
            dets = []
            if func in detections:
                dets = detections[func]
            f_tracked, f_lost = tracker.update(dets)
            self.stream._objdatas = f_tracked
            for item in f_tracked:
                track_id = item['track_id']
                if track_id not in self.vobj_pool:
                    self.vobj_pool[track_id] = func(self.stream)
                self.vobj_pool[track_id].update(item)
                tracked.append(self.vobj_pool[track_id])
            for item in f_lost:
                track_id = item['track_id']
                self.vobj_pool[track_id].update(None)
                lost.append(self.vobj_pool[track_id])
            # logger.info(f"tracking done")
        
        return tracked, lost

