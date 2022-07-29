import functools
from math import sqrt
from queue import Queue
from typing import Any, Dict, Callable, List, Tuple
from vqpy.utils.images import CropImage
from vqpy.utils.strings import longest_prefix_in

_vqpy_basefuncs: Dict[str, List[str]] = {}
_vqpy_libfuncs: Dict[str, Tuple[List[str], List[str], List[str], Callable]] = {}
def vqpy_func_logger(input_fields, output_fields, past_fields, specifications = None, required_length = -1):
    """Add function to log
    Args:
        input_fields: required fields in this frame.
        output_fields: provided fields.
        past_fields: required fields in past frames.
        specifications: preference of the function (TODO).
        required_length: the minimum track length for function to be useful.
    """
    # TODO: support specifications on classes of objects
    # TODO: support more accurate automatic selection of used functions
    def decorator(func : Callable):
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            if obj._track_length < required_length and required_length >= 0:
                return [None]
            return func(obj, *args, **kwargs)
        global _vqpy_libfuncs, _vqpy_basefuncs
        _vqpy_libfuncs[func.__name__] = (input_fields, output_fields, past_fields, wrapper)
        for index, field in enumerate(output_fields):
            if field not in _vqpy_basefuncs:
                _vqpy_basefuncs[field] = [func.__name__]
            else:
                _vqpy_basefuncs[field].append(func.__name__)
        return wrapper
    return decorator

@vqpy_func_logger(['tlbr'], ['bbox_velocity'], ['tlbr'], required_length=2)
def bbox_velocity(obj, tlbr):
    tlbr_c, tlbr_p = tlbr, obj.getv('tlbr', -2)
    center_c = (tlbr_c[:2] + tlbr_c[2:]) / 2
    center_p = (tlbr_p[:2] + tlbr_p[2:]) / 2
    tlbr_avg = (tlbr_c + tlbr_p) / 2
    scale = (tlbr_avg[3] - tlbr_avg[1]) / 1.5
    dcenter = (center_c - center_p) / scale * obj._ctx.fps
    v = sqrt(sum(dcenter * dcenter))
    return [v]

@vqpy_func_logger(['frame', 'tlbr'], ['image'], [], required_length=1)
def image_boundarycrop(obj, frame, tlbr):
    return [CropImage(frame, tlbr)]

@vqpy_func_logger(['image'], ['license_plate'], [], required_length=1)
def license_plate_lprnet(obj, image):
    from vqpy.models.lprnet import GetLP
    return [GetLP(image)]

@vqpy_func_logger(['image'], ['license_plate'], [], required_length=1)
def license_plate_openalpr(obj, image):
    from vqpy.models.openalpr import GetLP
    return [GetLP(image)]

@vqpy_func_logger(['tlbr'], ['coordinate'], [], required_length=1)
def coordinate_center(obj, tlbr):
    return (tlbr[:2] + tlbr[2:]) / 2

# @vqpy_logger
def infer(obj, attr: str, existing_fields: List[str], existing_pfields: List[str] = [], specifications = None):
    """Infer a undefined attribute based on provided fields and logged functions
    Args:
        obj (VObjBase): the vobject itself.
        attr (str): the attribute name to infer.
        existing_fields (List[str]): existing fields in this frame.
        existing_pfields (List[str], optional): existing fields in past frames. Defaults to [].
        specifications (Any, optional): hints for the infer. Defaults to None.
        Currently, we accept a set of strings as hints, and choose the functions having the longest prefix of the provided hints.

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
            eval = lambda x: longest_prefix_in(specifications[attr], x)
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
