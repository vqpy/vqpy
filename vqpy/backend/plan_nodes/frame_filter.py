from vqpy.backend.operator.frame_filter import VObjFrameFilter
from vqpy.backend.plan_nodes.base import AbstractPlanNode
from vqpy.frontend.query import QueryBase


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


def create_frame_filter_node(query_obj: QueryBase, input_node):
    output_node = input_node.set_next(VObjFrameFilterNode(filter_index=0))
    return output_node
