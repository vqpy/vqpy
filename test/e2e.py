# end to end testing, loitering example
from typing import Dict, List
import argparse
import os
import pickle

import vqpy
from vqpy.operator.detector.base import DetectorBase
from vqpy.class_names.coco import COCO_CLASSES
from vqpy.operator.detector import register


def make_parser():
    parser = argparse.ArgumentParser("VQPy end-to-end testing")
    parser.add_argument("--video", help="name of video to test")
    parser.add_argument("--path", help="path to precomputed object detection results")
    parser.add_argument(
        "--save_folder",
        default=None,
        help="the folder to save the final result",
    )
    return parser


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


class Person(vqpy.VObjBase):
    pass


class People_loitering_query(vqpy.QueryBase):
    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        REGION = [(550, 550), (1162, 400), (1720, 720), (1430, 1072), (600, 1073)]
        REGIONS = [REGION]

        filter_cons = {
            "__class__": lambda x: x == Person,
            "bottom_center": vqpy.query.continuing(
                condition=vqpy.query.utils.within_regions(REGIONS),
                duration=10,
                name="in_roi",
            ),
        }
        select_cons = {
            "track_id": None,
            "coordinate": lambda x: str(x),  # convert to string for
            # JSON serialization
            # name in vqpy.continuing + '_periods' can be used in select_cons.
            "in_roi_periods": None,
        }
        return vqpy.VObjConstraint(filter_cons, select_cons, filename="loitering")


args = make_parser().parse_args()
register(
    "fake-yolox",
    FakeYOLOX,
    os.path.join(
        args.path, f"{os.path.splitext(os.path.basename(args.video))[0]}_yolox.pkl"
    ),
    None
)

vqpy.launch(
    cls_name=COCO_CLASSES,
    cls_type={"person": Person},
    tasks=[People_loitering_query()],
    video_path=args.video,
    save_folder=args.save_folder,
    detector_name="fake-yolox",
)

# result saved to {save_folder}/{video_name}_{task_name}_{detector_name}.json