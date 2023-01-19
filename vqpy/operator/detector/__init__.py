"""
This folder (detector/) contains the VQPy detector interfaces.
All visible instances in this folder inherits (base.py).
"""

from vqpy.operator.detector.models.onnx.yolov4 import *   # noqa: F401,F403
from vqpy.operator.detector.models.onnx.faster_rcnn import *   # noqa: F401,F403
from vqpy.operator.detector.models.torch.yolox import *   # noqa: F401,F403
from vqpy.operator.detector.base import DetectorBase
import os
from loguru import logger


vqpy_detectors = {}


def register(detector_name, detector_type, model_filename):
    detector_name_lower = detector_name.lower()
    if detector_name_lower in vqpy_detectors:
        raise ValueError(f"Detector name {detector_name} is already in VQPy."
                         f"Please change another name to register.")
    vqpy_detectors[detector_name_lower] = (detector_type, model_filename)


def setup_detector(cls_names,
                   model_dir: str = None,
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
    logger.info(f"Detector {detector_name} is chosen!")
    detector_model_path = os.path.join(model_dir, model_filename)
    return detector_name, detector_type(model_path=detector_model_path)
