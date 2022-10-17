"""
This folder (detector/) contains the VQPy detector interfaces.
All visible instances in this folder inherits (base/detector.py).
"""

from .logger import vqpy_detectors
from ..base.detector import DetectorBase

# TODO: add automatic detector selection interface here


def setup_detector(cls_names, detector_model_path) -> DetectorBase:
    """setup a detector for video analytics
    cls_names: the detection class types of the required detector
    """
    for detector_type in vqpy_detectors:
        # Optional TODO: add ambiguous class match here
        if cls_names == detector_type.cls_names:
            return detector_type(model_path=detector_model_path)
