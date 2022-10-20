"""The logger for all VQPy detectors"""

vqpy_detectors = {}


def register(detector_name, detector_type, model_filename):
    detector_name_lower = detector_name.lower()
    if detector_name_lower in vqpy_detectors:
        raise ValueError(f"Detector name {detector_name} is already in VQPy."
                         f"Please change another name to register.")
    vqpy_detectors[detector_name_lower] = (detector_type, model_filename)
