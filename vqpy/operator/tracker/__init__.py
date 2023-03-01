"""
This folder (tracker/) contains the VQPy UNDERLYING tracker interfaces.
All visible instances in this folder inherits (base.py).
"""

from .byte_tracker import ByteTracker

vqpy_trackers = {}


def register(tracker_name, tracker):
    """Register a tracker"""
    tracker_name_lower = tracker_name.lower()
    if tracker_name_lower in vqpy_trackers:
        raise ValueError(f"Tracker name {tracker_name} is already in VQPy."
                         f"Please change another name to register.")
    vqpy_trackers[tracker_name_lower] = tracker


register("byte", ByteTracker)


def setup_ground_tracker(tracker_name: str = "byte", **kwargs):
    """Pickup appropriate ground-level tracker"""
    # TODO: add automatic tracker selection interface here
    if tracker_name not in vqpy_trackers:
        raise ValueError(f"Tracker name {tracker_name} hasn't been registered \
                          to VQPy.")
    tracker = vqpy_trackers[tracker_name]
    return tracker(**kwargs)
