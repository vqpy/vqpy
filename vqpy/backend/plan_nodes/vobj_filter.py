from vqpy.backend.operator.vobj_filter import VObjFilter
from vqpy.backend.plan_nodes.base import AbstractPlanNode
from vqpy.frontend.vobj.predicates import Predicate, IsInstance
from vqpy.frontend.query import QueryBase


class VObjFilterNode(AbstractPlanNode):

    def __init__(self,
                 predicate: Predicate,
                 filter_index):
        self.predicate = predicate
        self.filter_index = filter_index
        super().__init__()

    def to_operator(self, lauch_args: dict):
        return VObjFilter(
            prev=self.prev.to_operator(lauch_args),
            condition_func=self.predicate.generate_condition_function(),
            filter_index=self.filter_index
        )

    def __str__(self):
        return f"VObjFilterNode(predicate={self.predicate}, \n" \
               f"\tfilter_index={self.filter_index}), \n" \
               f"\tprev={self.prev.__class__.__name__}), \n"\
               f"\tnext={self.next.__class__.__name__})"


def create_vobj_class_filter_node(query_obj: QueryBase, input_node):
    frame_constraints = query_obj.frame_constraint()
    node = input_node

    vobjs = frame_constraints.get_vobjs()
    assert len(vobjs) == 1, "Only support one vobj in the predicate"
    vobj = list(vobjs)[0]
    node = node.set_next(
        VObjFilterNode(predicate=IsInstance(vobj), filter_index=0))
    return node


def create_vobj_filter_node_pred(predicate: Predicate, input_node):
    output_node = input_node
    output_node = output_node.set_next(
        VObjFilterNode(predicate=predicate, filter_index=0))
    return output_node


def create_vobj_filter_node_query(query_obj: QueryBase, input_node):
    predicate = query_obj.frame_constraint()

    output_node = create_vobj_filter_node_pred(predicate, input_node)

    return output_node
