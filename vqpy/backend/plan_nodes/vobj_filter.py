from vqpy.backend.operator.vobj_filter import VObjFilter
from vqpy.backend.plan_nodes.base import AbstractPlanNode
from vqpy.frontend.vobj.predicates import Predicate


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