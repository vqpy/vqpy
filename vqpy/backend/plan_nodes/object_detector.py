from vqpy.backend.operator.object_detector import ObjectDetector
from vqpy.backend.plan_nodes.base import AbstractPlanNode
from vqpy.frontend.query import QueryBase
from vqpy.frontend.vobj.predicates import Predicate

from typing import Optional, Set, Union


class ObjectDetectorNode(AbstractPlanNode):

    def __init__(self,
                 class_names: Union[str, Set[str]],
                 detector_name: Optional[str] = None,
                 detector_kwargs: dict = None):
        self.class_names = class_names
        self.detector_name = detector_name
        self.detector_kwargs = detector_kwargs \
            if detector_kwargs is not None else dict()
        super().__init__()

    def to_operator(self, launch_args: dict):
        return ObjectDetector(
            prev=self.prev.to_operator(launch_args),
            class_names=self.class_names,
            detector_name=self.detector_name,
            **self.detector_kwargs
        )


def create_object_detector_node(query_obj: QueryBase, input_node):
    frame_constraints = query_obj.frame_constraint()
    assert isinstance(frame_constraints, Predicate)
    vobjs = frame_constraints.get_vobjs()
    assert len(vobjs) == 1, "Only support one vobj in the predicate"
    vobj = list(vobjs)[0]
    class_names = vobj.class_name
    detector_name = vobj.object_detector
    detector_kwargs = vobj.detector_kwargs
    return input_node.set_next(
        ObjectDetectorNode(class_names=class_names,
                           detector_name=detector_name,
                           detector_kwargs=detector_kwargs)
    )
