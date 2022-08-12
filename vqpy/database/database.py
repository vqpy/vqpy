from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from ..base.interface import VObjBaseInterface

def _cons_add(u: Optional[Callable], v: Optional[Callable]):
    if u is None: return v
    if v is None: return u
    return lambda x: u(x) and v(x)

def _wrapped_call(f: Optional[Callable], x: Any):
    return f(x) if f is not None else x

def _filter(x: VObjBaseInterface, cond: Dict[str, Callable]) -> Optional[VObjBaseInterface]:
    for item, func in cond.items():
        it = x.getv(item)
        if it is None or not _wrapped_call(func, it):
            return None
    return x

class VObjConstraint:
    def __init__(self, filter_cons: Dict[str, Optional[Callable]] = {}, select_cons: Dict[str, Optional[Callable]] = {}, filename = "data.json"):
        self.filter_cons = filter_cons
        self.select_cons = select_cons
        self.filename = filename
    
    def __add__(self, other: VObjConstraint) -> VObjConstraint:
        # down + up
        ret = VObjConstraint(self.filter_cons, self.select_cons, self.filename)
        for key, cond in other.filter_cons.items():
            ret.filter_cons[key] = _cons_add(self.filter_cons.get(key, None), cond)
        return ret
    
    def apply(self, vobjs: List[VObjBaseInterface]) -> List[Dict]:
        filtered_vobjs = list(filter(None, list(map(lambda x: _filter(x, self.filter_cons), vobjs))))
        selected_datas = [{key: _wrapped_call(postproc, x.getv(key)) for key, postproc in self.select_cons.items()} for x in filtered_vobjs]
        return selected_datas

"""
argmin is not supported now as it requires information from multiple vobjects.

def vobj_argmin(tracks: List[VObjBaseInterface], func: Callable, args: List):
    def fill(a : List, b):
        return [x if x is not None else b for x in a]
    res, resv = None, None
    for x in tracks:
        xv = func(*fill(args, x))
        if res is None or xv < resv:
            res, resv = x, xv
    return res
"""