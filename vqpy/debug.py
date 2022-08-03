# for debugging only

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
        # logger.info(f'args={[tostr(x) for x in args]}, kwargs={tostr(kwargs)}')
        rets = func(*args, **kwargs)
        # logger.info(f'{rets}')
        return rets
    return wrapper