from vqpy.backend.operator.base import Operator
from vqpy.backend.frame import Frame
from vqpy.operator.tracker import vqpy_trackers
from typing import Optional


class Tracker(Operator):

    def __init__(self,
                 prev: Operator,
                 class_name: str,
                 tracker_name: Optional[str] = None,
                 filter_index: Optional[int] = None,
                 **tracker_kwargs,
                 ):
        """
        Tracker Operator.
        It uses the built-in tracker with name of {tracker_name}
        for tracking interested classes defined in {class_names}. It
        generates the `track_id` field in `vobj_data` on `frame`,
        which is the track id of the vobj.

        Args:
            prev (Operator): The previous operator instance.
            class_name: class name for tracking.
            tracker_name: Tracker name. e.g. "byte".
            filter_index: only track vobjs that are in the filtered_vobjs
             of the filter_index.
            tracker_kwargs: Keyword arguments for the tracker.
        """
        super().__init__(prev)
        tracker_name = tracker_name or "byte"
        self.tracker = vqpy_trackers[tracker_name](**tracker_kwargs)
        self.class_name = class_name
        self.filter_index = filter_index

    def _update_tracker(self, vobj_indexes, frame: Frame):
        if len(vobj_indexes) > 0:
            detections = []
            for index in vobj_indexes:
                vobj_data = frame.vobj_data[self.class_name][index]
                det = {k: vobj_data[k] for k in {"tlbr", "score"}}
                det["index"] = index
                detections.append(det)
            f_tracked, _ = self.tracker.update(frame.id, detections)
            for vobj in f_tracked:
                index = vobj['index']
                frame.vobj_data[self.class_name][index]['track_id'] = \
                    vobj["track_id"]
        return frame

    def next(self) -> Frame:
        if self.has_next():
            frame = self.prev.next()
            if self.filter_index is not None:
                if self.filter_index not in frame.filtered_vobjs:
                    raise ValueError("filter_index is not in filtered_vobjs")
                vobj_indexes = frame.filtered_vobjs[self.filter_index][self.class_name]
            else:
                if self.class_name not in frame.vobj_data:
                    return frame
                vobj_indexes = range(len(frame.vobj_data[self.class_name]))
            frame = self._update_tracker(vobj_indexes, frame)
        return frame
