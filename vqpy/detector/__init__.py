"""
This folder (detector/) contains the VQPy detector interfaces.
All visible instances in this folder inherits (base/detector.py).
"""
from .yolox import YOLOXDetector

# TODO: add automatic detector selection interface here

def setup_detector(device="cpu", fp16=False):
    return YOLOXDetector(device, fp16)