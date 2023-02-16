from vqpy.obj.frame import Frame
from abc import abstractmethod


class Operator:

    @abstractmethod
    def next(frame: Frame) -> Frame:
        pass
