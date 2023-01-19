from __future__ import annotations

from typing import Callable

from vqpy.obj.frame import FrameInterface
from vqpy.obj.vobj.base import VObjBaseInterface

VObjGeneratorType = Callable[[FrameInterface], VObjBaseInterface]
