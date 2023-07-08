import argparse
import vqpy
from typing import List, Tuple
import ast

from vqpy.frontend.vobj import VObjBase, vobj_property
from vqpy.frontend.query import QueryBase

NO_RISK = 0
WARNING = 1
ALARM = 2


def make_parser():
    parser = argparse.ArgumentParser("VQPy Demo!")
    parser.add_argument(
        "--path", help="path to video", default="./loitering.mp4"
    )
    parser.add_argument(
        "--save_folder",
        default=None,
        help="the folder to save the final result",
    )
    parser.add_argument(
        "--polygon",
        default=(
            "[(360, 367), (773, 267), (1143, 480), (951, 715), (399, 715)]"
        ),
        help="polygon to define the region of interest",
    )
    return parser


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
    @vobj_property(inputs={"bottom_center": 150})
    def in_region(self, values):
        bottom_centers = values["bottom_center"]

        def point_in_region(x):
            return x is not None and \
                vqpy.query.utils.within_regions(REGIONS)(x)
        nframes_warning = 60
        if all(list(map(point_in_region, bottom_centers))):
            return ALARM
        if all(list(map(point_in_region, bottom_centers[-nframes_warning:]))):
            return WARNING
        return NO_RISK


class People_loitering_query(QueryBase):
    def __init__(self) -> None:
        self.person = Person()

    def frame_constraint(self):
        return (self.person.in_region == ALARM) | (
            self.person.in_region == WARNING
        )

    def frame_output(self):
        return (self.person.center, self.person.tlbr, self.person.in_region)


if __name__ == "__main__":
    args = make_parser().parse_args()
    # Note that the loitering.mp4 is of shape 1274*720, therefore the polygon
    # points differs from that in the old example, with shape of 1920*1080.
    REGIONS = [ast.literal_eval(args.polygon)]

    query_executor = vqpy.init(
        video_path=args.path,
        query_obj=People_loitering_query(),
        verbose=True,
    )
    vqpy.run(query_executor, save_folder=args.save_folder)
