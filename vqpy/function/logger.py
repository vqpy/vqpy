"""The logger of VQPy libfunctions"""

import functools
from typing import Any, Callable, Dict, List, Tuple

_vqpy_basefuncs: Dict[str, List[str]] = {}
_vqpy_libfuncs: Dict[str, Tuple[List[str], List[str], List[str], Callable]] = {}
def vqpy_func_logger(input_fields, output_fields, past_fields, specifications = None, required_length = -1):
    """Add function to log
    Args:
        input_fields: required fields in this frame.
        output_fields: provided fields.
        past_fields: required fields in past frames.
        specifications: preference of the function.
        required_length: the minimum track length for function to be useful.
    """
    def decorator(func : Callable):
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            if obj._track_length < required_length and required_length >= 0:
                return [None]
            return func(obj, *args, **kwargs)
        _vqpy_libfuncs[func.__name__] = (input_fields, output_fields, past_fields, wrapper)
        for field in output_fields:
            if field not in _vqpy_basefuncs:
                _vqpy_basefuncs[field] = [func.__name__]
            else:
                _vqpy_basefuncs[field].append(func.__name__)
        return wrapper
    return decorator
