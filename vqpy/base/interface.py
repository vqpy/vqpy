"""Interfaces that requires implementations in impl/"""

from __future__ import annotations
from typing import Dict, List, Optional, Callable

from ..utils.video import FrameStream

from dataclasses import dataclass


class VObjBaseInterface(object):
    """The interface of VObject Base Class.
    The tracker is responsible to keep objects updated when the track is active
    """

    def __init__(self, frame: FrameInterface):
        self._frame = frame
        self._ctx = frame.ctx
        self._start_idx = frame.ctx.frame_id
        # Number of frames consecutively appears
        self._track_length = 0
        # Historic object data. TODO: shrink memory
        self._datas: List[Optional[Dict]] = []
        # List of @property instances
        self._registered_names: List[str] = []
        raise NotImplementedError

    def getv(self,
             attr: str,
             index: int = -1,
             specifications: Optional[Dict[str, str]] = None):
        """
        attr: attribute name.
        index: FRAMEID - Current FRAMEID - 1.
        specifications: optional dictionary for specifying models.
        Return: the value when applicable, and None otherwise.
        """
        raise NotImplementedError

    def update(self, data: Optional[Dict]):
        """Called once per frame by the tracker providing the object data"""
        if data is not None:
            self._datas.append(data.copy())
            self._track_length += 1
        else:
            self._datas.append(None)
            self._track_length = 0
        raise NotImplementedError

    def infer(self,
              attr: str,
              specifications: Optional[Dict[str, str]] = None):
        """A easy-to-use interface provided for usage of built-in functions"""
        raise NotImplementedError


class VObjConstraintInterface(object):
    """The interface of VObjConstraint class"""
    def __add__(self, other: VObjConstraintInterface):
        """merge constraints in the form subclass + superclass"""
        raise NotImplementedError

    def filter(self, objs: List[VObjBaseInterface]) -> List[VObjBaseInterface]:
        """filter the list of vobjects from the constraint"""
        raise NotImplementedError

    def apply(self, vobjs: List[VObjBaseInterface]) -> List[Dict]:
        """apply the constraint on a list of VObj instances"""
        raise NotImplementedError


class FrameInterface(object):
    def __init__(self, ctx: FrameStream):
        raise NotImplementedError

    def set_vobjs(self, vobjs: List[VObjBaseInterface]) -> None:
        raise NotImplementedError

    def update_vobjs(self,
                     vobj_type: VObjGeneratorType,
                     track_id: int,
                     data: Dict,
                     ) -> None:
        raise NotImplementedError

    def get_tracked_vobjs(self,
                          vobj_type: VObjGeneratorType,
                          ) -> List[VObjBaseInterface]:
        raise NotImplementedError

    def get_lost_vobjs(self,
                       vobj_type: VObjGeneratorType,
                       ) -> List[VObjBaseInterface]:
        raise NotImplementedError


@dataclass
class OutputConfig:
    """
    The config for Query output.
    :param output_frame_vobj_num: Default as False. whether to add the number
        of the filtered vobjs as an output for each frame. If true, we will
        generate a "vobj_num" field in each frame output.
    :param output_total_vobj_num: Default as False. whether to add the number
        of the filtered vobjs as an output for the whole video. If true, we
         will generate a "total_vobj_num" field in the output.
    """
    output_frame_vobj_num: bool = False
    output_total_vobj_num: bool = False


VObjGeneratorType = Callable[[FrameInterface], VObjBaseInterface]
