from vqpy.backend.frame import Frame
from abc import abstractmethod


class Operator:

    def __init__(self, prev=None) -> None:
        self.prev = prev

    def has_next(self) -> bool:
        # video reader and stateful projector need to overwrite.
        if self.prev:
            return self.prev.has_next()
        else:
            raise NotImplementedError

    @abstractmethod
    def next() -> Frame:
        pass
