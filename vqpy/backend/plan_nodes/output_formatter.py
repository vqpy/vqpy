from vqpy.backend.operator.output_formatter import FrameOutputFormatter
from vqpy.backend.plan_nodes.base import AbstractPlanNode
from vqpy.frontend.query import QueryBase
from vqpy.frontend.vobj.property import Property

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


def create_frame_output_formatter(query_vobj: QueryBase,
                                  input_node):
    frame_output = query_vobj.frame_output()
    if isinstance(frame_output, Property):
        frame_output = [frame_output]
    property_mappings = dict()
    vobj_name_mappings = dict()

    # todo: support multiple vobjs
    filter_index = 0
    vobj = frame_output[0].get_vobjs()
    assert len(vobj) == 1, "Only support one vobj for vobj_property."
    vobj = list(vobj)[0]
    class_name = vobj.class_name
    name = vobj.name
    vobj_name_mappings[filter_index] = name
    property_mappings[filter_index] = {
        class_name: [prop.name for prop in frame_output]
    }
    frame_output_formatter = FrameOutputNode(
        filter_index_to_class_name_to_property_names=property_mappings,
        filter_index_to_vobj_name=vobj_name_mappings
    )
    return input_node.set_next(frame_output_formatter)
