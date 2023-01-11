"""VObjBase implementation"""

from typing import Dict, List, Optional, Set

from ..base.interface import VObjBaseInterface
from ..function import infer
from ..utils.video import FrameStream


class VObjBase(VObjBaseInterface):
    """The VObject Base Class.
    The tracker is responsible to keep objects updated when the track is active
    """

    def __init__(self, ctx: FrameStream):
        self._ctx = ctx
        self._start_idx = ctx.frame_id
        self._track_length = 0
        self._datas: List[Optional[Dict]] = []
        self._registered_names: Set[str] = set()
        self._registered_cross_vobj_names: Dict[str, ] = {}
        self._working_infers: List[str] = []
        # NOTE: now @property instances are stored in the order of __dir__()
        for instance_name in self.__dir__():
            instance = getattr(self, instance_name)
            if instance_name[0] != '_' and callable(instance):
                try:
                    instance()
                except TypeError:
                    pass

    def _get_fields(self):
        return list(self._datas[0].keys()) + \
            list(self._registered_names) + self._ctx.output_fields

    def _get_pfields(self):
        return list(self._datas[0].keys()) + [x for x in self._registered_names
                                              if hasattr(self, '__state_' + x)]

    def getv(self,
             attr: str,
             index: int = -1,
             specifications: Optional[Dict[str, str]] = None):
        """
        NOTE: Note the order in the following checking.
        Infer an attribute of the object from:
        (1) __static (2) _datas (3) __state_ (4) _ctx.output_fields
        (5) __record_ (6) _registered_names (7) vqpy.infer

        attr: attribute name.
        index: FRAMEID - Current FRAMEID - 1.
        specifications: optional dictionary for specifying models.
        # TODO: expand specification to include model parameters and etc.

        return: the value when applicable, and None otherwise.

        For paramterized getv, write UDFs to compute the required properties.

        __static_ can be used to store properties that are not time-related.
        e.g. color of object, aggregated properties
        """
        # patchwork to support __static_
        if hasattr(self, '__static_' + attr):
            return getattr(self, '__static_' + attr)
        idx = self._ctx.frame_id + index + 1 - self._start_idx
        if idx < 0 or idx > len(self._datas):
            return None
        elif (idx < len(self._datas) and
              self._datas[idx] is not None and
              attr in self._datas[idx]):
            return self._datas[idx][attr]
        elif index == -1:
            if attr in self._ctx.output_fields:
                return getattr(self._ctx, attr)
            elif (hasattr(self, '__record_' + attr) and
                  getattr(self, '__index_' + attr) == self._ctx.frame_id):
                return getattr(self, '__record_' + attr)
            elif attr in self._registered_names:
                return getattr(self, attr)()    # TODO: test if used at all
            else:
                assert len(self._datas) > 0
                self._working_infers.append(attr)
                # Avoid circular calls when inferring by remove working infers
                nfields = [x for x in self._get_fields()
                           if x not in self._working_infers]
                pfields = self._get_pfields()
                value = infer(self, attr, nfields, pfields, specifications)
                self._working_infers.pop()
                # following handles built-in case like __class__
                if value is None:
                    value = getattr(self, attr, None)
                return value
        elif hasattr(self, '__state_' + attr):
            values = getattr(self, '__state_' + attr)
            idx = index + self._ctx.frame_id - getattr(self, '__index_' + attr)
            if 0 < -idx <= len(values):
                return values[idx]
            else:
                return None

    def update(self, data: Optional[Dict]):
        """Update data this frame to object"""
        if data is not None:
            self._datas.append(data.copy())
            self._track_length += 1
        else:
            self._datas.append(None)
            self._track_length = 0
        for method_name in self._registered_names:
            # properties updated here
            getattr(self, method_name)()

    def infer(self,
              attr: str,
              specifications: Optional[Dict[str, str]] = None):
        """A easy-to-use interface provided for usage of built-in functions"""
        return infer(self, attr, self._get_fields(), self._get_pfields(),
                     specifications)
