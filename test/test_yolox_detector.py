from vqpy.operator.detector import vqpy_detectors
from PIL import Image
import numpy as np
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(current_dir, "resources/")


def test_yolox_detector_basic():
    detector_type, model_weights_path = vqpy_detectors["yolox"]
    detector = detector_type(model_path=model_weights_path, device="cpu")

    input_data = np.asarray(Image.open(os.path.join(resource_dir, "cat.jpg")))

    output = detector.inference(input_data)

    expected_tlbr = np.array([3.7286086, 378.80682, 2796.176, 3368.2278])
    expected_score = 0.95613
    expected_class_id = 15
    assert len(output) == 1
    assert np.allclose(output[0]["tlbr"], expected_tlbr, rtol=0, atol=1)
    assert np.allclose(output[0]["score"], expected_score, rtol=0, atol=0.01)
    assert output[0]["class_id"] == expected_class_id
