"""The features for easy coding in VQPy"""

import functools
from typing import Callable, Dict, List, Optional

from ..base.interface import VObjBaseInterface


def property():
    """
    @property decorator should ALWAYS appear as the FIRST decorator.
    Ensures a method of an VObj is computed exactly once per frame.
    When having incoming updates, call all functions with @property decorators.
    TODO: add more features for @property
    NOTE: This is never called directly hence overriding is not a problem.
    """
    def decorator(func: Callable):
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
                self._registered_names.add(func.__name__)
                return None
        return wrapper
    return decorator


def stateful(length: int = 0):
    """
    @stateful decorator stores the return value into `__state_` field.
    A function decorated withthis also should be decorated with @property.
    TODO: we can merge @property and @stateful.
    length: number of frames to reserve the value, by default store all values.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self: VObjBaseInterface, *args, **kwargs):
            attr = '__state_' + func.__name__
            if not hasattr(self, attr):
                setattr(self, attr, [])
            new_value = func(self, *args, **kwargs)
            values: List = getattr(self, attr)
            if length > 0 and len(values) == length:
                values = values[1:]
            values.append(new_value)
            setattr(self, attr, values)
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
                    if it in local_map:
                        local_map[it] += 1
                    else:
                        local_map[it] = 1
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
Now we do not support it. If this type of property is required, please
compute it using the database.

# TODO: make it more clear and locate in the right place
def access_data(cond: Dict[str, Callable]):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self: VObjBaseInterface):
            return func(self, vobj_filter(self._ctx._objdatas, cond))
        return wrapper
    return decorator
"""


# for declaring data dependencies and retrieving data from other vobjs
# currently don't register dependencies since execution sequence is fixed
# cross_vobj_property only needs to provide the required list of properties
# of vobjs
def cross_vobj_property(
        vobj_type=None, vobj_num="ALL", vobj_input_fields=None
        ):
    """Decorator for cross-object property computation.

    Wrapper for cross-object property computation.

    The property function being decorated should accept two arguments:
    `self` and `cross_vobj_arg` (positional).
    During execution, VQPy will pass `cross_vobj_arg`, a list of properties of
    VObjs of specified type, to the property function being decorated.

    `cross_vobj_arg` has structure:
    `List[Tuple(property1, property2, ...) for vobj1, Tuple for vobj2, ...]`,
    where property1, property2 are values of property names listed in
    `vobj_input_fields`, in order; vobj1, vobj2 are VObjs of type `vobj_type`.

    Attributes:
    vobj_type: VObjGeneratorType
        type of VObj to retrieve
    vobj_num: int
        number of VObjs to retrieve
    vobj_input_fields: List[str]
        list of names of properties to retrieve from VObjs
    """
    # vobj_num defaults to "ALL" for now
    # other possible options could be user-specified number
    def wrap(func: Callable):
        @functools.wraps(func)
        def wrapped_func(
            self: VObjBaseInterface,
            cross_vobj_arg: Optional[List] = None
        ):
            # parameter cross_vobj_arg has default value None to maintain
            # the "same" interface with @property upon being called directly
            # this compatibility is used in VObjBase.__init__ at instance()
            if len(self._datas) > 0:
                vidx = '__record_' + func.__name__
                aidx = '__index_' + func.__name__
                if getattr(self, aidx, None) == self._ctx.frame_id:
                    return getattr(self, vidx)
                else:
                    value = func(self, cross_vobj_arg)
                    setattr(self, vidx, value)
                    setattr(self, aidx, self._ctx.frame_id)
                    return value
            elif func.__name__ not in self._registered_cross_vobj_names:
                # initialization
                # register function name, required VObj type and fields
                self._registered_cross_vobj_names[func.__name__] = \
                    (vobj_type, vobj_input_fields)
                return None
        return wrapped_func
    return wrap
