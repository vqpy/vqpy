from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable, Any
from vqpy.funcutils import vqpy_logger
from vqpy.video_loader import FrameStream
from vqpy.functions import _vqpy_libfuncs, infer
import functools

def property():
    """
    Ensures a method of an VObj is computed exactly once per frame.
    When having incoming updates, call all functions with @property decorators.
    TODO: add more features for @property
    """
    def decorator(func : Callable):
        @functools.wraps(func)
        def wrapper(self: VObjBase, *args, **kwargs):
            if len(self._datas) > 0:
                vidx = '__record_' + func.__name__
                aidx = '__index_' + func.__name__
                if getattr(self, aidx, None) == self._ctx.frame_id:
                    return getattr(self, vidx)
                else:
                    value = func(self, *args, **kwargs)
                    setattr(self, vidx, value)
                    setattr(self, aidx, self._ctx.frame_id)
                    return value
            elif func.__name__ not in self._registered_names:
                # initialization
                self._registered_names.append(func.__name__)
                return None
        return wrapper
    return decorator

def stateful(length: int = 0):
    """
    TODO: we can merge @property to @stateful.
    ALL @stateful FUNCTIONS SHOULD BE @property FUNCTIONS.
    Store historic values of a function into '__state_' field.
    length:     the number of frames to reserve the value, by default store all values.
    """
    def decorator(func : Callable):
        @functools.wraps(func)
        def wrapper(self: VObjBase, *args, **kwargs):
            attr = '__state_' + func.__name__
            if not hasattr(self, attr):
                setattr(self, attr, [])
            values: List = getattr(self, attr)
            new_value = func(self, *args, **kwargs)
            if length > 0 and len(values) == length:
                values = values[1:]
            values.append(new_value)
            return values[-1]
        return wrapper
    return decorator

def worker(cond: Dict):
    """
    Fetch the detection results to functional arguments.
    cond: which results to fetch
    """
    # TODO: complete worker definition

class VObjBase(object):
    # When the vobject is active, keep it updated
    
    def __init__(self, ctx: FrameStream):
        self._ctx = ctx
        self._start_idx = ctx.frame_id
        self._track_length = 0
        self._datas: List[Optional[Dict]] = []
        self._registered_names: List[str] = []          # List of @property instances
        for instance_name in self.__dir__():
            instance = getattr(self, instance_name)
            if instance_name[0] != '_' and callable(instance):
                try:
                    instance()
                except TypeError:
                    pass
        # print(self.__class__, self._registered_names)
    
    def getv(self, attr: str, index: int = -1):
        """
        If you want to use parameterized getv, write one more function to handle it instead.
        Infer a attribute of the object FROM _datas, __state_ and functions (user-defined first).
        attr: attribute name.
        index: FRAMEID - Current FRAMEID - 1.
        return: the value when applicable, and None otherwise.
        When an attribute has an non-property user-defined function/value of the same name in the object, we will return the UDF.
        NOTE: THE ORDER IN index==-1 checking is important.
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
            elif attr in _vqpy_libfuncs:
                assert(len(self._datas) > 0)
                fields = list(self._datas[0].keys()) + self._registered_names + self._ctx.output_fields
                return infer(self, attr, fields)
            elif hasattr(self, attr):
                return getattr(self, attr)
            else:
                return None
        elif hasattr(self, '__state_' + attr):
            values = getattr(self, '__state_' + attr)
            idx = index + self._ctx.frame_id - getattr(self, '__index_' + attr)
            if 0 < -idx <= len(values):
                return values[idx]
            else:
                return None
    
    @vqpy_logger
    def update(self, data: Optional[Dict]):
        # TODO: it is possible to not store all the items
        if data is not None:
            self._datas.append(data.copy())
            self._track_length += 1
        else:
            self._datas.append(None)
            self._track_length = 0
        for method_name in self._registered_names:
            getattr(self, method_name)()

VObjGeneratorType = Callable[[FrameStream], VObjBase]
