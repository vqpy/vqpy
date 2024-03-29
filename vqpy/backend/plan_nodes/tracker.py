from vqpy.backend.operator.tracker import Tracker
from vqpy.backend.plan_nodes.base import AbstractPlanNode

from vqpy.frontend.query import QueryBase
from vqpy.frontend.vobj.predicates import Predicate

from typing import Optional


class TrackerNode(AbstractPlanNode):

    def __init__(self,
                 class_name: str,
                 filter_index: Optional[int] = None,
                 tracker_name: str = "byte",
                 ):
        self.class_name = class_name
        self.filter_index = filter_index
        self.tracker_name = tracker_name
        super().__init__()

    def to_operator(self, launch_args: dict):
        # fps is for byte tracker
        return Tracker(
            prev=self.prev.to_operator(launch_args),
            class_name=self.class_name,
            filter_index=self.filter_index,
            tracker_name=self.tracker_name,
            fps=launch_args["fps"],
        )

    def __str__(self):
        return f"TrackerNode(class_name={self.class_name}, \n" \
            f"\tfilter_index={self.filter_index}, \n" \
            f"\ttracker_name={self.tracker_name}), \n" \
            f"\tprev={self.prev.__class__.__name__}), \n"\
            f"\tnext={self.next.__class__.__name__})"


def create_tracker_node(query_obj: QueryBase, input_node):
    frame_constraints = query_obj.frame_constraint()
    assert isinstance(frame_constraints, Predicate)
    vobjs = frame_constraints.get_vobjs()
    assert len(vobjs) == 1, "Only support one vobj in the predicate"
    vobj = list(vobjs)[0]
    class_name = vobj.class_name
    return input_node.set_next(
        TrackerNode(class_name=class_name)
    )
