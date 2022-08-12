from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from ..base.interface import VObjBaseInterface

def _cons_add(lside: Optional[Callable], rside: Optional[Callable]):
    if lside is None:
        return rside
    if rside is None:
        return lside
    return lambda x: lside(x) and rside(x)

def _wrapped_call(func: Optional[Callable], args: Any):
    return func(args) if func is not None else args

def _filter(obj: VObjBaseInterface, cond: Dict[str, Callable]) -> Optional[VObjBaseInterface]:
    for item, func in cond.items():
        it = obj.getv(item)
        if it is None or not _wrapped_call(func, it):
            return None
    return obj

class VObjConstraint:
    """The constraint on VObj instances, helpful when applying queries"""

    def __init__(self, filter_cons: Dict[str, Optional[Callable]] = {}, select_cons: Dict[str, Optional[Callable]] = {}, filename = "data"):
        """Initialize a VObj constraint instances

        Args:
            filter_cons (Dict[str, Optional[Callable]], optional):
                the filter constraints, an item pair (key, cond) denotes the property 'key' of
                the VObj instance should satisfies cond(key) is True. When cond is None, the
                instance should satisfies key is True. Defaults to {}.
            select_cons (Dict[str, Optional[Callable]], optional):
                the select constraints, an item pair (key, proc) denotes select property 'key'
                of the VObj instance and then apply 'proc' on it. When cond is None, we do the
                identity transformation. Defaults to {}.
            filename (str, optional): the saved json name. Defaults to "data" (save in "data.json").
        """
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
        """apply the constraint on a list of VObj instances"""
        # TODO: optimize the procedure
        filtered = list(filter(None, list(map(lambda x: _filter(x, self.filter_cons), vobjs))))
        selected = [{key: _wrapped_call(postproc, x.getv(key))
                     for key, postproc in self.select_cons.items()} for x in filtered]
        return selected

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