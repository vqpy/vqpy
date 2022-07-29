
import functools
from typing import Callable, Dict, List, Optional, Tuple
from vqpy.objects import VObjBase

def _filter(x: VObjBase, cond: Dict[str, Callable]) -> Optional[VObjBase]:
    for item, func in cond.items():
        it = x.getv(item)
        if it is None or not func(it):
            return None
    return x

def vobj_filter(tracks: List[VObjBase], cond: Dict[str, Callable]) -> List[VObjBase]:
    """works like WHERE clause in SQL
    Select some fields of the list of vobjects
    Result is returned as list of dictionary
    """
    return list(filter(None, list(map(lambda x: _filter(x, cond), tracks))))

def vobj_select(tracks: List[VObjBase], cond: Dict[str, Callable]) -> List[Dict]:
    """works like SELECT clause in SQL
    Select some fields of the list of vobjects
    Result is returned as list of dictionary
    """
    return [{key: postproc(x.getv(key)) for key, postproc in cond.items()} for x in tracks]

def vobj_argmin(tracks: List[VObjBase], func: Callable, args: List):
    def fill(a : List, b):
        return [x if x is not None else b for x in a]
    res, resv = None, None
    for x in tracks:
        xv = func(*fill(args, x))
        if res is None or xv < resv:
            res, resv = x, xv
    return res

# TODO: make it more clear and locate in the right place
def access_data(cond: Dict[str, Callable]):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self: VObjBase):
            return func(self, vobj_filter(self._ctx._objdatas, cond))
        return wrapper
    return decorator