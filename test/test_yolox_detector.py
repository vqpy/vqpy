# import yolox
# import numpy as np
# from yolox.exp.build import get_exp
# model_path = "/home/yang/sources/vqpy/vqpy/operator/detector/weights/yolox_x.pth"
# exp = get_exp(None, "yolox_x")
# model = exp.get_model()
# model.eval()
# ckpt = torch.load(model_path, map_location="cpu")
# model.load_state_dict(ckpt["model"])

# data = np.random.randint()
# model()

from vqpy.operator.detector import vqpy_detectors
from vqpy.class_names.coco import COCO_CLASSES
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

if __name__ == "__main__":
    test_yolox_detector()

    