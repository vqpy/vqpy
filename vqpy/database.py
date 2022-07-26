
from typing import Callable, Dict, List, Optional
from vqpy.objects import VObjBase

class QueryBase:
    def apply(self, tracks: List[VObjBase]) -> List[Dict]:
        pass

def _filter(x: VObjBase, cond: Dict[str, Callable]) -> Optional[VObjBase]:
    for item, func in cond.items():
        it = x.getv(item)
        if it is None or not func(it):
            return None
    return x

def vobj_filter(tracks: List[VObjBase], cond: Dict[str, Callable]) -> List[VObjBase]:
    return list(filter(None, list(map(lambda x: _filter(x, cond), tracks))))

def vobj_select(tracks: List[VObjBase], cond: Dict[str, Callable]) -> List[Dict]:
    return [{key: postproc(x.getv(key)) for key, postproc in cond.items()} for x in tracks]