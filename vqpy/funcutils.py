from typing import Callable
import functools
from loguru import logger

def vqpy_logger(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        def tostr(args):
            ret = f'{args}'
            ret = ret.replace('        ', ' ')
            ret = ret.replace('\n', ' ')
            if len(ret) > 100: ret = ret[:100] + '...) '
            return ret
        # logger.info(f'CALL {func.__module__}::{func.__name__} with parameter:')
        # logger.info(f'args={tostr(args)}, kwargs={tostr(kwargs)}')
        return func(*args, **kwargs)
    return wrapper

import numpy as np

def tlbr_to_xyah(tlbr):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    ret = np.asarray(tlbr).copy()
    ret[2:] -= ret[:2]
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret

@vqpy_logger
def CropImage(img, tlbr, ext=0):
    _, _, w, h = tlbr_to_xyah(tlbr)
    if tlbr[0] >= -w * 0.2 and tlbr[1] >= -h * 0.2 and \
       tlbr[2] <= img.shape[1] + w * 0.2 and tlbr[3] <= img.shape[0] + h * 0.2:
        minx, miny, maxx, maxy = tlbr
        minx = int(max(minx - w * ext / 2, 0))
        miny = int(max(miny - h * ext / 2, 0))
        maxx = int(min(maxx + w * ext / 2, img.shape[1]))
        maxy = int(min(maxy + h * ext / 2, img.shape[0]))
        return img[miny:maxy+1, minx:maxx+1, :]
    return None