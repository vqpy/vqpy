"""
This folder (detector/) contains the VQPy detector interfaces.
All visible instances in this folder inherits (base.py).
"""

from vqpy.operator.detector.models.onnx.yolov4 import Yolov4Detector
from vqpy.operator.detector.models.onnx.faster_rcnn import FasterRCNNDdetector
from vqpy.operator.detector.models.torch.yolox import YOLOXDetector
from vqpy.operator.detector.base import DetectorBase
import os
from loguru import logger

dir_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_DETECTOR_WEIGHTS_DIR = os.path.join(dir_path, "weights/")
vqpy_detectors = {}


def register(detector_name, detector_type, model_weights_path):
    detector_name_lower = detector_name.lower()
    if detector_name_lower in vqpy_detectors:
        raise ValueError(f"Detector name {detector_name} is already in VQPy."
                         f"Please change another name to register.")
    vqpy_detectors[detector_name_lower] = (detector_type, model_weights_path)


register("yolox", YOLOXDetector,
         os.path.join(DEFAULT_DETECTOR_WEIGHTS_DIR, "yolox_x.pth"))
register("faster_rcnn", FasterRCNNDdetector,
         os.path.join(DEFAULT_DETECTOR_WEIGHTS_DIR, "FasterRCNN-10.onnx"))
register("yolov4", Yolov4Detector,
         os.path.join(DEFAULT_DETECTOR_WEIGHTS_DIR, "yolov4.onnx"))


def setup_detector(cls_names,
                   detector_name: str = None,
                   ) -> (str, DetectorBase):
    """setup a detector for video analytics
    cls_names: the detection class types of the required detector
    """
    if detector_name:
        if detector_name not in vqpy_detectors:
            raise ValueError(f"Detector name of {detector_name} hasn't been"
                             f"registered to VQPy")
        detector_type, model_weights_path = vqpy_detectors[detector_name]

    else:
        # TODO: add automatic detector selection interface here
        for detector_name in vqpy_detectors:
            # Optional TODO: add ambiguous class match here
            detector_type, model_weights_path = vqpy_detectors[detector_name]
            if cls_names == detector_type.cls_names:
                print(f"Detector {detector_name} has been selected!")
                break
    logger.info(f"Detector {detector_name} is chosen!")
    if not os.path.exists(model_weights_path):
        raise ValueError(f"Cannot find weights path {model_weights_path}")
    return detector_name, detector_type(model_path=model_weights_path)
