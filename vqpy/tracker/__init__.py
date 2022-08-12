"""
This folder (tracker/) contains the VQPy UNDERLYING tracker interfaces.
All visible instances in this folder inherits (base/ground_tracker.py).
"""
from ..utils.video import FrameStream

from .byte_tracker import ByteTracker

# TODO: add automatic tracker selection interface here

def setup_ground_tracker(ctx: FrameStream):
    return ByteTracker(ctx)
