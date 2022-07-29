from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Callable, Any
from vqpy.video_loader import FrameStream
from vqpy.functions import _vqpy_libfuncs, infer
import functools

def property():
    """
    @property decorator should ALWAYS appear as the FIRST decorator.
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
    @stateful decorator stores the return value into `__state_` field.
    If a function is decorated with @stateful, it also should be decorated with @property.
    TODO: we can merge @property to @stateful.
    length: the number of frames to reserve the value, by default store all values.
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

def postproc(params: Dict):
    """
    Run a set of postprocessing functions for the function's result.
    Input the parameters in a dictionary of form {process_name: setting}
    Currently only `majority` is supported, it has a integer parameter N,
      denoting that we return the majority vote of the result in last N frames.
    """
    if 'majority' in params:
        size = params['majority']
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                name = '__postproc_majority_' + func.__name__
                if not hasattr(self, name):
                    setattr(self, name, [])
                values = getattr(self, name)
                values.append(func(self, *args, **kwargs))
                if len(values) > size:
                    values = values[1:]
                local_map = {}
                for it in values:
                    if it in local_map: local_map[it] += 1
                    else: local_map[it] = 1
                ret = (None, 0)
                for it, v in local_map.items():
                    if it is not None and v > ret[1]:
                        ret = (it, v)
                return ret[0]
            return wrapper
        return decorator
    else:
        raise NotImplementedError

class VObjBase(object):
    # When the vobject is active, keep it updated
    
    def __init__(self, ctx: FrameStream):
        self._ctx = ctx
        self._start_idx = ctx.frame_id
        self._track_length = 0
        self._datas: List[Optional[Dict]] = []
        self._registered_names: List[str] = []          # List of @property instances
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
            elif attr in _vqpy_libfuncs:
                assert(len(self._datas) > 0)
                self._working_infers.append(attr)
                # When inferring instances, we remove the currently working infers from fields, to avoid circular calling
                nfields = [x for x in self._get_fields() if x not in self._working_infers]
                pfields = self._get_pfields()
                value = infer(self, attr, nfields, pfields, specifications)
                self._working_infers.pop()
                return value
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
    
    def infer(self, attr: str, specifications: Optional[Dict[str, str]] = None):
        """A easy-to-use interface provided to user to use functions in built-in functions"""
        return infer(self, attr, self._get_fields(), self._get_pfields(), specifications)

VObjGeneratorType = Callable[[FrameStream], VObjBase]
