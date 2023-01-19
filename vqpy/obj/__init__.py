from __future__ import annotations

from typing import Callable

from vqpy.impl.frame import FrameInterface
from vqpy.impl.vobj_base import VObjBaseInterface

VObjGeneratorType = Callable[[FrameInterface], VObjBaseInterface]
