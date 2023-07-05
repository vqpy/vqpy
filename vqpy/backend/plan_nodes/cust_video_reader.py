from vqpy.backend.plan_nodes.base import AbstractPlanNode
from vqpy.backend.operator import CustomizedVideoReader


def create_customized_video_reader_node(operator: CustomizedVideoReader):
    node_name = operator.__class__.__name__ + "Node"
    CustVideoReaderNode = type(node_name,
                               (AbstractPlanNode,),
                               {"to_operator": lambda self, lauch_args: operator})
    return CustVideoReaderNode()
