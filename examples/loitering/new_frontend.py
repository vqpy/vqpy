import argparse
import vqpy
from typing import List, Tuple
import ast

from vqpy.frontend.vobj import VObjBase, vobj_property
from vqpy.frontend.query import QueryBase

NO_RISK = "no_risk"
WARNING = "warning"
ALARM = "alarm"

TOLERANCE = 10


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
    parser.add_argument(
        "--time_warning",
        default=4,
        help="time to trigger warning",
    )
    parser.add_argument(
        "--time_alarm",
        default=10,
        help="time to trigger alarm",
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

    @vobj_property(inputs={"in_region_time": 0})
    def loitering(self, values):
        cur_in_region_time = values["in_region_time"]
        if cur_in_region_time >= TIME_ALARM:
            return ALARM
        if cur_in_region_time >= TIME_WARNING:
            return WARNING
        return NO_RISK

    @vobj_property(inputs={"in_region_frames": 0, "fps": 0, "in_region": 0})
    def in_region_time(self, values):
        cur_in_region_frames = values["in_region_frames"]
        fps = values["fps"]
        if not values["in_region"]:
            return 0
        return round(cur_in_region_frames / fps, 2)

    @vobj_property(
        inputs={"in_region": TOLERANCE, "in_region_frames": TOLERANCE}
    )
    def in_region_frames(self, values):
        """
        Return the number of frames that the person is in region continuously.
        If the person is out of region for longer than TOLERANCE, return 0.
        If the person is out of region cur frame and within TORLERANCE,
          the in_region_frames is the same as that of last frame.
        If the person is untracked and tracked again within in TORLENCE frames,
          the time is accumulated. Otherwise, the in_region_frames is 0.
        """
        in_region_values = values["in_region"]
        # Get the last valid in_region_frames. If person is lost and tracked
        # again, the in_region_frames for lost frames are None.
        last_valid_in_region_frames = 0
        for value in reversed(values["in_region_frames"]):
            if value is not None:
                last_valid_in_region_frames = value
                break
        this_in_region = in_region_values[-1]
        if this_in_region:
            return last_valid_in_region_frames + 1
        else:
            # The person is out of region for longer than TOLERANCE frames
            if last_valid_in_region_frames == in_region_values[0]:
                return 0
            else:
                return last_valid_in_region_frames

    @vobj_property(inputs={"bottom_center": 0})
    def in_region(self, values):
        bottom_center = values["bottom_center"]
        if bottom_center is not None and vqpy.query.utils.within_regions(
            REGIONS
        )(bottom_center):
            return True
        return False


class People_loitering_query(QueryBase):
    def __init__(self) -> None:
        self.person = Person()

    def frame_constraint(self):
        return self.person.in_region_time > 0

    def frame_output(self):
        return (
            self.person.track_id,
            self.person.center,
            self.person.tlbr,
            self.person.loitering,
            self.person.in_region_time,
        )


if __name__ == "__main__":
    args = make_parser().parse_args()
    # Note that the loitering.mp4 is of shape 1274*720, therefore the polygon
    # points differs from that in the old example, with shape of 1920*1080.
    REGIONS = [ast.literal_eval(args.polygon)]
    TIME_WARNING = args.time_warning
    TIME_ALARM = args.time_alarm

    query_executor = vqpy.init(
        video_path=args.path,
        query_obj=People_loitering_query(),
        verbose=True,
        output_per_frame_results=True,
    )
    vqpy.run(query_executor, save_folder=args.save_folder)
