import argparse
import vqpy
from typing import List, Tuple
import numpy as np

from vqpy.backend.plan import Planner, Executor
from vqpy.frontend.vobj import VObjBase, vobj_property
from vqpy.frontend.query import QueryBase


def make_parser():
    parser = argparse.ArgumentParser("VQPy Demo!")
    parser.add_argument("--path", help="path to video")
    parser.add_argument(
        "--save_folder",
        default=None,
        help="the folder to save the final result",
    )
    return parser


REGION = [(550, 550), (1162, 400), (1720, 720), (1430, 1072), (600, 1073)]
REGIONS = [REGION]


class Person(VObjBase):
    def __init__(self) -> None:
        self.class_name = "person"
        self.object_detector = "yolox"
        self.detector_kwargs = {"device": "gpu"}
        super().__init__()

    @vobj_property(inputs={"tlbr": 0})
    def bottom_center(self, values) -> List[Tuple[int, int]]:
        tlbr = values["tlbr"]
        x = (tlbr[0] + tlbr[2]) / 2
        y = tlbr[3]
        return [(x, y)]

    @vobj_property(inputs={"tlbr": 0})
    def center(self, values):
        tlbr = values["tlbr"]
        return (tlbr[:2] + tlbr[2:]) / 2

    # TODO: continuing not yet supported, workaround
    @vobj_property(inputs={"bottom_center": 9})
    def in_region(self, values):
        bottom_centers = values["bottom_center"]
        in_region = np.all(
            [
                bottom_center is not None
                and vqpy.query.utils.within_regions(REGIONS)(bottom_center)
                for bottom_center in bottom_centers
            ]
        )
        return in_region


class People_loitering_query(QueryBase):
    def __init__(self) -> None:
        self.person = Person()

    def frame_constraint(self):
        return self.person.in_region == True  # noqa: E712

    def frame_output(self):
        return {
            "center": self.person.center,
        }


if __name__ == "__main__":
    args = make_parser().parse_args()
    planner = Planner()
    launch_args = {"video_path": args.path}
    root_plan_node = planner.parse(People_loitering_query())
    planner.print_plan(root_plan_node)
    executor = Executor(root_plan_node, launch_args)
    result = executor.execute()

    for frame in result:
        print(frame.id)
        for person_idx in frame.filtered_vobjs[0]["person"]:
            print(frame.vobj_data["person"][person_idx])
