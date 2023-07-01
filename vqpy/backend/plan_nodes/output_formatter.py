from vqpy.backend.operator.output_formatter import FrameOutputFormatter
from vqpy.backend.plan_nodes.base import AbstractPlanNode


from typing import Dict, List


class FrameOutputNode(AbstractPlanNode):

    def __init__(self,
                 filter_index_to_class_name_to_property_names:
                 Dict[int, Dict[str, List[str]]],
                 filter_index_to_vobj_name: Dict[int, str]
                 ):
        self.props_mapping = filter_index_to_class_name_to_property_names
        self.vobj_names_mapping = filter_index_to_vobj_name
        super().__init__()

    def to_operator(self, lauch_args: dict):
        return FrameOutputFormatter(
            prev=self.prev.to_operator(lauch_args),
            filter_index_to_class_name_to_property_names=self.props_mapping,
            filter_index_to_vobj_name=self.vobj_names_mapping)

    def __str__(self):
        return f"VObjFrameOutputNode(props_mapping={self.props_mapping}), \n"\
            f"\tprev={self.prev.__class__.__name__}), \n" \
            f"\tnext={self.next.__class__.__name__})"