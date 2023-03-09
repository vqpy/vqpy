from vqpy.backend.operator.base import Operator
from vqpy.backend.frame import Frame
from typing import Callable, Tuple


class FrameFilter(Operator):
    def __init__(self,
                 prev: Operator,
                 condition_func: Callable[[Frame], bool]
                 ):
        self.condition_func = condition_func
        self.current_frame = None
        super().__init__(prev)

    def has_next(self) -> bool:
        if self.current_frame:
            return True
        while self.prev.has_next():
            frame = self.prev.next()
            if self.condition_func(frame):
                self.current_frame = frame
                return True
        return False

    def next(self) -> Frame:
        if self.has_next():
            frame = self.current_frame
            self.current_frame = None
            return frame
        else:
            raise StopIteration


class VObjFrameFilter(FrameFilter):
    def __init__(self, prev: Operator, vobj_filter_index: int = 0):
        def condition_func(frame: Frame):
            if vobj_filter_index > len(frame.filtered_vobjs):
                raise ValueError(f"Vobj filter index {vobj_filter_index} "
                                 f"out of range")
            filtered_vobjs = frame.filtered_vobjs[vobj_filter_index]
            return any([bool(filtered_vobjs[key]) for key in filtered_vobjs])
        super().__init__(prev, condition_func)


class FrameRangeFilter(FrameFilter):
    def __init__(self, prev: Operator, frame_id_range: Tuple[int, int]):
        def condition_func(frame: Frame):
            return frame.id in range(*frame_id_range)
        super().__init__(prev, condition_func)
