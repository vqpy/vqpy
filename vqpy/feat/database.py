import functools
from typing import Callable, Dict, List, Optional
from vqpy.base.interface import VObjBaseInterface

def _filter(x: VObjBaseInterface, cond: Dict[str, Callable]) -> Optional[VObjBaseInterface]:
    for item, func in cond.items():
        it = x.getv(item)
        if it is None or not func(it):
            return None
    return x

def vobj_filter(tracks: List[VObjBaseInterface], cond: Dict[str, Callable]) -> List[VObjBaseInterface]:
    """works like WHERE clause in SQL
    Select some fields of the list of vobjects
    Result is returned as list of dictionary
    """
    return list(filter(None, list(map(lambda x: _filter(x, cond), tracks))))

def vobj_select(tracks: List[VObjBaseInterface], cond: Dict[str, Callable]) -> List[Dict]:
    """works like SELECT clause in SQL
    Select some fields of the list of vobjects
    Result is returned as list of dictionary
    """
    return [{key: postproc(x.getv(key)) for key, postproc in cond.items()} for x in tracks]

def vobj_argmin(tracks: List[VObjBaseInterface], func: Callable, args: List):
    def fill(a : List, b):
        return [x if x is not None else b for x in a]
    res, resv = None, None
    for x in tracks:
        xv = func(*fill(args, x))
        if res is None or xv < resv:
            res, resv = x, xv
    return res
