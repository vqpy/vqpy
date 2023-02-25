from vqpy.operator.detector.base import DetectorBase
from vqpy.class_names.coco import COCO_CLASSES
from typing import Dict, List
import pickle
from vqpy.operator.detector import register
from pathlib import Path


class FakeYOLOX(DetectorBase):
    cls_names = COCO_CLASSES
    output_fields = ["tlbr", "score", "class_id"]

    def __init__(self, model_path, **detector_kwargs):
        # load file
        with open(model_path, "rb") as file:
            self.detection_result = pickle.load(file)
        self.frame_id = 1

    def inference(self, img) -> List[Dict]:
        # read from file
        outputs = self.detection_result[self.frame_id]
        self.frame_id += 1
        return outputs


video_name = "pedestrian_10s"
resource_dir = Path(Path(__file__).parent.parent.parent, "resources/")

precomputed_path = (resource_dir / f"{video_name}_yolox.pkl").as_posix()
print(resource_dir)
print(precomputed_path)

register("fake_yolox", FakeYOLOX, precomputed_path, None)
