"""The features for easy coding in VQPy"""

import functools
from typing import Any, Callable, Dict, List

from ..base.interface import VObjBaseInterface

def property():
    """
    @property decorator should ALWAYS appear as the FIRST decorator.
    Ensures a method of an VObj is computed exactly once per frame.
    When having incoming updates, call all functions with @property decorators.
    TODO: add more features for @property
    NOTE: This is never called directly hence overriding is not a problem.
    """
    def decorator(func : Callable):
        @functools.wraps(func)
        def wrapper(self: VObjBaseInterface, *args, **kwargs):
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
        def wrapper(self: VObjBaseInterface, *args, **kwargs):
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
            def wrapper(self: VObjBaseInterface, *args, **kwargs):
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

"""
This feature is to support the computation of cross-object property.
Now we do not support it. If this type of property is required, please compute it using the database.

# TODO: make it more clear and locate in the right place
def access_data(cond: Dict[str, Callable]):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self: VObjBaseInterface):
            return func(self, vobj_filter(self._ctx._objdatas, cond))
        return wrapper
    return decorator
"""