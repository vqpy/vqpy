from abc import ABC, abstractmethod
from vqpy.frontend.vobj.vobj import VObjBase
from vqpy.frontend.vobj.predicates import IsInstance


class QueryBase(ABC):

    @abstractmethod
    def frame_constraint(self):
        pass

    @abstractmethod
    def frame_output(self):
        pass

    def internal_frame_constraint(self):
        cons = self.frame_constraint()
        if isinstance(cons, VObjBase):
            return IsInstance(cons)
