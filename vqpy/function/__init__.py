"""
This folder (function/) contains the VQPy lib functions.
The (expected) only visible interface of this folder is `infer`.
"""

from queue import Queue
from typing import Any, Dict, List

from ..utils.strings import longest_prefix_in as _default_metric

from .functions import *
from .logger import _vqpy_basefuncs, _vqpy_libfuncs

# TODO: formally define the format of `specifications`, supporting more accurate
# automatic selection, applicable condition for functions, etc.

def infer(obj, attr: str, existing_fields: List[str], existing_pfields: List[str] = [], specifications = None):
    """Infer a undefined attribute based on provided fields and logged functions
    Args:
        obj (VObjBase): the vobject itself.
        attr (str): the attribute name to infer.
        existing_fields (List[str]): existing fields in this frame.
        existing_pfields (List[str], optional): existing fields in past frames. Defaults to [].
        specifications (Any, optional): hints for the infer. Defaults to None.
        Currently, we accept a set of strings as hints, and choose the functions having the longest
        prefix of the provided hints.

    Returns:
        The inferred attribute value.
    """
    if attr not in _vqpy_basefuncs:
        return None
    if specifications is None:
        specifications = {}
    data: Dict[str, Any] = {}
    waitlist = [attr]
    calls = []
    q: Queue = Queue()
    q.put(attr)
    while not q.empty():
        attr = q.get()
        eval = None
        if attr in specifications:
            eval = lambda x: _default_metric(specifications[attr], x)
        INF = 1e4
        best, bscore = None, -INF**2
        for name in _vqpy_basefuncs[attr]:
            input_fields, output_fields, past_fields, _ = _vqpy_libfuncs[name]
            alive = True
            for field in past_fields:
                if field not in existing_pfields:
                    alive = False
                    break
            if not alive:
                continue
            for field in input_fields:
                if field in waitlist:
                    alive = False
                    break
            if not alive:
                continue
            score = 0 if eval is None else eval(name) * INF
            for field in input_fields:
                if field not in existing_fields:
                    score -= 5
            for field in output_fields:
                if field in waitlist:
                    score += 1
            if score > bscore:
                best = name
                bscore = score
        if best is None:
            return None
        input_fields, output_fields, _, _ = _vqpy_libfuncs[best]
        for field in input_fields:
            if field not in existing_fields:
                waitlist.append(field)
                q.put(field)
        calls.append(best)
    calls.reverse()
    # logger.info(f'Infer execution order: {calls}')
    for name in calls:
        input_fields, output_fields, _, func = _vqpy_libfuncs[name]
        # print(input_fields)
        args = [obj] + [obj.getv(x) if x in existing_fields else data[x] for x in input_fields]
        # this is the required args format
        outputs = func(*args)
        for i, value in enumerate(outputs):
            data[output_fields[i]] = value

    return data[waitlist[0]]
