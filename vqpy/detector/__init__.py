"""
This folder (detector/) contains the VQPy detector interfaces.
All visible instances in this folder inherits (base/detector.py).
"""

from .logger import vqpy_detectors
from ..base.detector import DetectorBase
import os


def setup_detector(cls_names,
                   detector_model_dir: str = None,
                   detector_name: str = None,
                   ) -> DetectorBase:
    """setup a detector for video analytics
    cls_names: the detection class types of the required detector
    """
    if detector_name:
        if detector_name not in vqpy_detectors:
            raise ValueError(f"Detector name of {detector_name} hasn't been"
                             f"registered to VQPy")
        detector_type, model_filename = vqpy_detectors[detector_name]

    else:
        # TODO: add automatic detector selection interface here
        for detector_name in vqpy_detectors:
            # Optional TODO: add ambiguous class match here
            detector_type, model_filename = vqpy_detectors[detector_name]
            if cls_names == detector_type.cls_names:
                print(f"Detector {detector_name} has been selected!")
                break
    detector_model_path = os.path.join(detector_model_dir, model_filename)
    return detector_type(model_path=detector_model_path)
