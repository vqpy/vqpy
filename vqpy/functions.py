import functools
from math import sqrt
from queue import Queue
from typing import Dict, Callable, List, Tuple
from vqpy.funcutils import *

_vqpy_basefuncs: Dict[str, List[str]] = {}
_vqpy_libfuncs: Dict[str, Tuple[List[str], List[str], List[str], Callable]] = {}
def vqpy_func_logger(input_fields, output_fields, stateful_input_fields, specifications = None):
    # generate a property from other properties
    # TODO: support specifications on classes of objects
    # TODO: support more accurate automatic selection of used functions
    # TODO: merge input_fields and stateful_input_fields in this backend
    def decorator(func : Callable):
        global _vqpy_libfuncs, _vqpy_basefuncs
        _vqpy_libfuncs[func.__name__] = (input_fields, output_fields, stateful_input_fields, func)
        for index, field in enumerate(output_fields):
            if field not in _vqpy_basefuncs:
                _vqpy_basefuncs[field] = [func.__name__]
            else:
                _vqpy_basefuncs[field].append(func.__name__)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

@vqpy_func_logger([], ['bbox_velocity'], ['tlbr'])
def bbox_velocity(obj):
    if obj._track_length < 2: return [None]
    tlbr_c, tlbr_p = obj.getv('tlbr', -1), obj.getv('tlbr', -2)
    center_c = (tlbr_c[:2] + tlbr_c[2:]) / 2
    center_p = (tlbr_p[:2] + tlbr_p[2:]) / 2
    tlbr_avg = (tlbr_c + tlbr_p) / 2
    scale = (tlbr_avg[3] - tlbr_avg[1]) / 1.5
    dcenter = (center_c - center_p) / scale * obj._ctx.fps
    v = sqrt(sum(dcenter * dcenter))
    return [v]

@vqpy_func_logger(['frame', 'tlbr'], ['license_plate'], [])
def license_plate(obj, frame, tlbr):
    from vqpy.models.lprnet import GetLP
    if obj._track_length < 1: return [None]
    img = CropImage(frame, tlbr)
    return [GetLP(img)]

@vqpy_logger
def infer(obj, attr: str, existing_fields: List[str]):
    if attr not in _vqpy_libfuncs:
        return None
    data = {}
    waitlist = [attr]
    calls = []
    q = Queue()
    q.put(attr)
    while not q.empty():
        attr = q.get()
        best, bscore = None, -1e4
        for name in _vqpy_basefuncs[attr]:
            input_fields, output_fields, _, _ = _vqpy_libfuncs[name]
            alive = True
            for field in input_fields:
                if field in waitlist:
                    alive = False
                    break
            if not alive:
                break
            score = 0
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
            # resolve attribute failed
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
