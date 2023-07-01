from vqpy.backend.operator.frame_filter import VObjFrameFilter
from vqpy.backend.plan_nodes.base import AbstractPlanNode


class VObjFrameFilterNode(AbstractPlanNode):

    def __init__(self, filter_index: int = 0):
        self.filter_index = filter_index
        super().__init__()

    def to_operator(self, lauch_args: dict):
        return VObjFrameFilter(prev=self.prev.to_operator(lauch_args),
                               vobj_filter_index=self.filter_index)

    def __str__(self):
        return f"VObjFrameFilterNode(filter_index={self.filter_index}), \n"\
            f"\tprev={self.prev.__class__.__name__}), \n" \
            f"\tnext={self.next.__class__.__name__})"