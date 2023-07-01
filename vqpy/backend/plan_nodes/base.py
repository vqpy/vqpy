from abc import abstractmethod


class AbstractPlanNode:

    def __init__(self):
        self.next = None
        self.prev = None

    def set_prev(self, plan_node):
        '''
        Set the plan_node as the current node's child node. The current node
        will consume or depends on the child node.
        return the child node
        '''
        self.prev = plan_node
        plan_node.next = self
        return plan_node

    def get_prev(self):
        return self.prev

    def set_next(self, plan_node):
        self.next = plan_node
        plan_node.prev = self
        return self.next

    def get_next(self):
        return self.next

    @abstractmethod
    def to_operator(self, lauch_args: dict):
        pass

    def __str__(self):
        return f"PlanNode({self.__class__.__name__},\n" \
                f"\tprev={self.prev.__class__.__name__},\n" \
                f"\tnext={self.next.__class__.__name__})"