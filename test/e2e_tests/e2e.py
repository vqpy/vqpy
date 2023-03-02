# end to end testing, loitering example
from typing import Dict, List
import pickle
import json
import warnings

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
def compare(result_path, expected_result_path, tolerance=0.1):
    with open(result_path, "r") as f:
        result = json.load(f)
    with open(expected_result_path, "r") as f:
        expected = json.load(f)

    # allow some mismatch
    assert abs(len(result) - len(expected)) < len(expected) * tolerance, (
        f"result length error, {len(result)} present, expect {len(expected)},"
        f" difference {abs(len(result) - len(expected))} > tolerance"
        f" {len(expected)*tolerance}"
    )

    mismatch_cnt = 0
    result_it = 0
    expected_it = 0
    while result_it < len(result) and expected_it < len(expected):
        if result[result_it]["frame_id"] < expected[expected_it]["frame_id"]:
            result_it += 1
            mismatch_cnt += 1
        elif result[result_it]["frame_id"] > expected[expected_it]["frame_id"]:
            expected_it += 1
            mismatch_cnt += 1
        else:
            assert (
                result[result_it]["data"] == expected[expected_it]["data"]
            ), (
                f"frame {result[result_it]['frame_id']} has data"
                f" {result[result_it]['data']}, expect"
                f" {expected[expected_it]['data']}"
            )
            result_it += 1
            expected_it += 1

    mismatch_cnt += (len(result) - result_it) + (len(expected) - expected_it)
    assert mismatch_cnt < len(expected) * tolerance, (
        f"mismatched {mismatch_cnt} frames >"
        f" {len(expected)*tolerance} tolerance"
    )

    if mismatch_cnt > 0:
        warnings.warn(
            f"compare result {result_path} to expected {expected_result_path},"
            f" mismatched {mismatch_cnt} frames"
        )
