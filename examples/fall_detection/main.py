import sys
import os

from vqpy.backend.plan import Planner, Executor
from vqpy.frontend.vobj import VObjBase, vobj_property
from vqpy.frontend.query import QueryBase

import torch
import numpy as np
import argparse

# importing pose detection models
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "detect")
)
from PoseEstimateLoader import SPPE_FastPose  # noqa: E402
from ActionsEstLoader import TSSTG  # noqa: E402


def make_parser():
    parser = argparse.ArgumentParser("VQPy Demo!")
    parser.add_argument("--path", help="path to video")
    parser.add_argument(
        "--save_folder",
        default=None,
        help="the folder to save the final result",
    )
    parser.add_argument(
        "--model_dir",
        default=None,
        help=(
            "folder containing pretrained model"
            " fast_res50_256x192.pth and tsstg-model.pth"
        ),
    )
    return parser


class Person(VObjBase):
    pose_model = None
    action_model = None

    def __init__(self) -> None:
        self.class_name = "person"
        self.object_detector = "yolox"
        self.detector_kwargs = {"device": "gpu"}
        super().__init__()

    @vobj_property(inputs={"tlbr": 0})
    def center(self, values):
        tlbr = values["tlbr"]
        return (tlbr[:2] + tlbr[2:]) / 2

    @vobj_property(
        inputs={"tlbr": 0, "image": 0, "frame_width": 0, "frame_height": 0}
    )
    def keypoints(self, values):
        image = values["image"]
        tlbr = values["tlbr"]
        frame_width = int(values["frame_width"])
        frame_height = int(values["frame_height"])
        return Person.pose_model.predict(
            image, torch.tensor(np.array([tlbr])), frame_width, frame_height
        )

    @vobj_property(
        inputs={"keypoints": 30 - 1, "frame_width": 0, "frame_height": 0}
    )
    def pose(self, values) -> str:
        keypoints_list = values["keypoints"]
        frame_width = values["frame_width"]
        frame_height = values["frame_height"]
        if any(keypoints is None for keypoints in keypoints_list):
            return "unknown"
        pts = np.array(keypoints_list, dtype=np.float32)
        out = Person.action_model.predict(pts, [frame_width, frame_height])
        action_name = Person.action_model.class_names[out[0].argmax()]
        return action_name


class FallDetection(QueryBase):
    def __init__(self) -> None:
        self.person = Person()

    def frame_constraint(self):
        return self.person.pose == "Fall Down"

    def frame_output(self):
        return {"center": self.person.center}


if __name__ == "__main__":
    args = make_parser().parse_args()
    model_dir = args.model_dir
    pose_model = SPPE_FastPose(
        backbone="resnet50",
        device="cpu",
        weights_file=os.path.join(
            os.path.abspath(model_dir), "fast_res50_256x192.pth"
        ),
    )
    action_model = TSSTG(
        weight_file=os.path.join(os.path.abspath(model_dir), "tsstg-model.pth")
    )
    Person.pose_model = pose_model
    Person.action_model = action_model
    planner = Planner()
    launch_args = {"video_path": args.path}
    root_plan_node = planner.parse(FallDetection())
    planner.print_plan(root_plan_node)
    executor = Executor(root_plan_node, launch_args)
    result = executor.execute()

    for frame in result:
        print(frame.id)
        for person_idx in frame.filtered_vobjs[0]["person"]:
            print(frame.vobj_data["person"][person_idx])
