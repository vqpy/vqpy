"""
This folder (detector/) contains the VQPy detector interfaces.
All visible instances in this folder inherits (base.py).
"""

from vqpy.operator.detector.models.onnx.yolov4 import Yolov4Detector
from vqpy.operator.detector.models.onnx.faster_rcnn import FasterRCNNDdetector
from vqpy.operator.detector.models.torch.yolox import YOLOXDetector
from vqpy.operator.detector.base import DetectorBase
import os
from typing import Optional, Tuple
from loguru import logger
import torch.hub

dir_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_DETECTOR_WEIGHTS_DIR = os.path.join(dir_path, "weights/")

if not os.path.exists(DEFAULT_DETECTOR_WEIGHTS_DIR):
    os.makedirs(DEFAULT_DETECTOR_WEIGHTS_DIR)

vqpy_detectors = {}

# Important TODO: import the model only when the model is used
# Otherwise we have to clone yolox model to support it


def register(detector_name,
             detector_type,
             model_weights_path,
             model_weights_url=None):

    detector_name_lower = detector_name.lower()
    if detector_name_lower in vqpy_detectors:
        raise ValueError(f"Detector name {detector_name} is already in VQPy."
                         f"Please change another name to register.")
    vqpy_detectors[detector_name_lower] = (detector_type,
                                           model_weights_path,
                                           model_weights_url)


yolox_path = os.path.join(DEFAULT_DETECTOR_WEIGHTS_DIR, "yolox_x.pth")
yolox_url = "https://github.com/Megvii-BaseDetection/YOLOX/" + \
    "releases/download/0.1.1rc0/yolox_x.pth"

register("yolox", YOLOXDetector, yolox_path, yolox_url)

faster_rnnn_path = os.path.join(DEFAULT_DETECTOR_WEIGHTS_DIR,
                                "FasterRCNN-10.onnx")
register("faster_rcnn", FasterRCNNDdetector, faster_rnnn_path, None)

yolov4_path = os.path.join(DEFAULT_DETECTOR_WEIGHTS_DIR, "yolov4.onnx")
register("yolov4", Yolov4Detector, yolov4_path, None)


def setup_detector(cls_names,
                   detector_name: Optional[str] = None,
                   detector_args: Optional[dict] = dict()
                   ) -> Tuple[str, DetectorBase]:
    """setup a detector for video analytics
    cls_names: the detection class types of the required detector
    """
    if detector_name:
        detector_name = detector_name.lower()
        if detector_name not in vqpy_detectors:
            raise ValueError(f"Detector name of {detector_name} hasn't been"
                             f"registered to VQPy")
        detector_type, weights_path, url = vqpy_detectors[detector_name]
    else:
        # TODO: add automatic detector selection interface here
        for detector_name in vqpy_detectors:
            # Optional TODO: add ambiguous class match here
            detector_type, weights_path, url = vqpy_detectors[detector_name]
            if detector_type.cls_names.issuperset(cls_names):
                print(f"Detector {detector_name} has been selected!")
                break
    logger.info(f"Detector {detector_name} is chosen!")
    if not os.path.exists(weights_path):
        if url is not None:
            torch.hub.download_url_to_file(url, weights_path)
        else:
            raise ValueError(f"Cannot find weights path {weights_path}")
    return detector_name, detector_type(model_path=weights_path,
                                        **detector_args)
