from typing import Callable, Dict, List, Optional

from ..base.interface import VObjBaseInterface
from ..function import infer
from ..utils.video import FrameStream


class VObjBase(VObjBaseInterface):
    # When the vobject is active, keep it updated
    
    def __init__(self, ctx: FrameStream):
        self._ctx = ctx
        self._start_idx = ctx.frame_id
        self._track_length = 0
        self._datas: List[Optional[Dict]] = []
        self._registered_names: List[str] = []
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
        return list(self._datas[0].keys()) + self._registered_names + self._ctx.output_fields
    def _get_pfields(self):
        return list(self._datas[0].keys()) + [x for x in self._registered_names if hasattr(self, '__state_' + x)]
    
    def getv(self, attr: str, index: int = -1, specifications: Optional[Dict[str, str]] = None):
        """
        NOTE: Note the order in the following checking.
        Infer an attribute of the object from:
        (1) _datas (2) __state_ (3) _ctx.output_fields (4) __record_
        (5) _registered_names (6) vqpy.infer
        
        attr: attribute name.
        index: FRAMEID - Current FRAMEID - 1.
        specifications: optional dictionary for specifying models.
        # TODO: expand specification definition to include model parameters and etc.
        
        return: the value when applicable, and None otherwise.
        
        For paramterized getv, write other functions to compute the required properties.
        """
        idx = self._ctx.frame_id + index + 1 - self._start_idx
        if idx < 0 or idx > len(self._datas):
            return None
        elif idx < len(self._datas) and self._datas[idx] is not None and attr in self._datas[idx]:
            return self._datas[idx][attr]
        elif index == -1:
            if attr in self._ctx.output_fields:
                return getattr(self._ctx, attr)
            elif hasattr(self, '__record_' + attr) and getattr(self, '__index_' + attr) == self._ctx.frame_id:
                return getattr(self, '__record_' + attr)
            elif attr in self._registered_names:
                return getattr(self, attr)()
            else:
                assert(len(self._datas) > 0)
                self._working_infers.append(attr)
                # When inferring instances, we remove the currently working infers from fields, to avoid circular calling
                nfields = [x for x in self._get_fields() if x not in self._working_infers]
                pfields = self._get_pfields()
                value = infer(self, attr, nfields, pfields, specifications)
                self._working_infers.pop()
                return value if value is not None else getattr(self, attr, None) # handle built-in case like __class__
        elif hasattr(self, '__state_' + attr):
            values = getattr(self, '__state_' + attr)
            idx = index + self._ctx.frame_id - getattr(self, '__index_' + attr)
            if 0 < -idx <= len(values):
                return values[idx]
            else:
                return None
    
    def update(self, data: Optional[Dict]):
        if data is not None:
            self._datas.append(data.copy())
            self._track_length += 1
        else:
            self._datas.append(None)
            self._track_length = 0
        for method_name in self._registered_names:
            getattr(self, method_name)()
    
    def infer(self, attr: str, specifications: Optional[Dict[str, str]] = None):
        """A easy-to-use interface provided to user to use functions in built-in functions"""
        return infer(self, attr, self._get_fields(), self._get_pfields(), specifications)

VObjGeneratorType = Callable[[FrameStream], VObjBase]
