# end to end testing, loitering example
from typing import Dict, List
import pickle
import json

from vqpy.operator.detector.base import DetectorBase
from vqpy.class_names.coco import COCO_CLASSES


class FakeYOLOX(DetectorBase):
    cls_names = COCO_CLASSES
    output_fields = ["tlbr", "score", "class_id"]

    def __init__(self, model_path):
        # load file
        with open(model_path, "rb") as file:
            self.detection_result = pickle.load(file)
        self.frame_id = 1

    def inference(self, img) -> List[Dict]:
        # read from file
        outputs = self.detection_result[self.frame_id]
        self.frame_id += 1
        return outputs


# compare results
def compare(result_path, expected_path):
    with open(result_path, "r") as f:
        result = json.load(f)
    with open(expected_path, "r") as f:
        expected = json.load(f)
    assert len(result) == len(
        expected
    ), f"result length error, {len(result)} present, expect {len(expected)}"
    for i in range(len(result)):
        assert (
            result[i]["frame_id"] == expected[i]["frame_id"]
        ), f"{i}th entry has frame_id {result[i]['frame_id']}, expect {expected[i]['frame_id']}"
        assert (
            result[i]["data"] == expected[i]["data"]
        ), f"frame {result[i]['frame_id']} has data {result[i]['data']}, expect {expected[i]['data']}"
