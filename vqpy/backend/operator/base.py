from vqpy.obj.frame_new import Frame
from abc import abstractmethod


class Operator:

    @abstractmethod
    def next() -> Frame:
        pass
