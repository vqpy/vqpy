"""
This folder (tracker/) contains the VQPy UNDERLYING tracker interfaces.
All visible instances in this folder inherits (base.py).
"""
from ..utils.video import FrameStream

from .byte_tracker import ByteTracker


def setup_ground_tracker(ctx: FrameStream):
    """Pickup appropriate ground-level tracker"""
    # TODO: add automatic tracker selection interface here
    return ByteTracker(ctx)
