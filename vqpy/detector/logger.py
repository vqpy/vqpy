"""The logger for all VQPy detectors"""

vqpy_detectors = []


def register(detector_type):
    vqpy_detectors.append(detector_type)
