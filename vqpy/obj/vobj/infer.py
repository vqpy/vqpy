from queue import Queue
from typing import List, Dict, Any

# import the library to include default builtin functions
from vqpy.property_lib import *  # noqa: F401,F403
from vqpy.property_lib.wrappers import _vqpy_basefuncs, _vqpy_libfuncs


def longest_prefix_in(b: str, a: str):
    """Return the length of the longest prefix of b that appears in a"""
    # return the longest prefix of b that appears in a (for best match)
    left, right = 0, len(b)
    while left < right:
        mid = (left + right + 1) >> 1
        if b[:mid] in a:
            left = mid
        else:
            right = mid - 1
    return left


def infer(obj,
          attr: str,
          existing_fields: List[str],
          existing_pfields: List[str] = [],
          specifications=None):
    """Infer a undefined attribute with provided fields and logged functions
    Args:
        obj (VObjBase): the vobject itself.
        attr (str): the attribute name to infer.
        existing_fields (List[str]): existing fields in this frame.
        existing_pfields (List[str], optional):
            existing fields in past frames. Defaults to [].
        specifications (Any, optional): hints for the infer. Defaults to None.
        Currently, we accept a set of strings as hints, and choose the function
        having the longest prefix of the provided hints.

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

        def evaluate(x):
            if attr in specifications:
                return longest_prefix_in(specifications[attr], x)
            else:
                return 0

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
            score = evaluate(name) * INF
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
        args = [obj] + [obj.getv(x) if x in existing_fields else
                        data[x] for x in input_fields]
        # this is the required args format
        outputs = func(*args)
        for i, value in enumerate(outputs):
            data[output_fields[i]] = value

    return data[waitlist[0]]
