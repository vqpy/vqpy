import argparse
import vqpy
from typing import Tuple
import ast

from vqpy.frontend.vobj import VObjBase, vobj_property
from vqpy.frontend.query import QueryBase


TOLERANCE = 10


def make_parser():
    parser = argparse.ArgumentParser("VQPy Demo!")
    parser.add_argument(
        "--path", help="path to video", default="./retail-store.mp4"
    )
    parser.add_argument(
        "--save_folder",
        default="results",
        help="the folder to save the final result",
    )
    parser.add_argument(
        "--polygon",
        # default="[(164,122),(434,77),(478,122),(393,247),(516,334),(241,431)]",
        default="[(156,130),(434,77),(478,122),(393,247),(516,334),(241,431)]",
        help="polygon to define the area of interest",
    )
    return parser


class Person(VObjBase):
    def __init__(self) -> None:
        self.class_name = "person"
        self.object_detector = "yolox"
        self.detector_kwargs = {"device": "gpu"}
        super().__init__()

    @vobj_property(inputs={"tlbr": 0})
    def bottom_center(self, values) -> Tuple[int, int]:
        tlbr = values["tlbr"]
        x = (tlbr[0] + tlbr[2]) / 2
        y = tlbr[3]
        return (x, y)

    @vobj_property(inputs={"tlbr": 0})
    def center(self, values):
        tlbr = values["tlbr"]
        return (tlbr[:2] + tlbr[2:]) / 2

    @vobj_property(inputs={"in_region_frames": 0, "fps": 0})
    def in_region_time(self, values):
        cur_in_region_frames = values["in_region_frames"]
        fps = values["fps"]
        if not cur_in_region_frames:
            return 0
        return round(cur_in_region_frames / fps, 2)

    @vobj_property(inputs={"in_region": 0, "in_region_frames": TOLERANCE})
    def in_region_frames(self, values):
        """
        Return the number of frames that the person is in region continuously.
        If the person is out of region for longer than TOLERANCE, return 0.
        If the person is out of region cur frame and within TORLERANCE,
          the in_region_frames is the same as that of last frame.
        If the person is untracked and tracked again within in TORLENCE frames,
          the time is accumulated. Otherwise, the in_region_frames is 0.
        """
        in_region = values["in_region"]
        # Get the last valid in_region_frames. If person is lost and tracked
        # again, the in_region_frames for lost frames are None.
        last_valid_in_region_frames = 0
        hist_in_region_frames = reversed(values["in_region_frames"][:-1])
        for value in hist_in_region_frames:
            if value is not None:
                last_valid_in_region_frames = value
                break
        if in_region:
            return last_valid_in_region_frames + 1
        else:
            # The person is out of region for longer than TOLERANCE frames
            if last_valid_in_region_frames == values["in_region_frames"][0]:
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

    @vobj_property(inputs={"tlbr": 1})
    def track_id_dummy(self, values):
        return 1

    @vobj_property(inputs={"tlbr": 5, "fps": 0})
    def velocity(self, values):
        """
        Return the velocity of the person in the last 1 second.
        """
        # use the top center as the center
        tlbrs = values["tlbr"]

        # locate the start and end point with tlbr not None
        for start, start_tlbr in enumerate(tlbrs):
            if start_tlbr is not None:
                break

        for i, end_tlbr in enumerate(reversed(tlbrs)):
            # history is 10 tlbrs
            end = 4 - i
            if end == start:
                return None
            if end_tlbr is not None:
                break

        # distance between the top center of start tlbr and end tlbr
        start_x = (start_tlbr[0] + start_tlbr[2]) / 2
        start_y = start_tlbr[1]
        end_x = (end_tlbr[0] + end_tlbr[2]) / 2
        end_y = end_tlbr[1]
        distance = ((start_x - end_x) ** 2 + (start_y - end_y) ** 2) ** 0.5
        velocity = distance / (end - start) * values["fps"]
        return round(velocity, 2)


class QueueAnalysis(QueryBase):
    def __init__(self) -> None:
        self.person = Person()

    def frame_constraint(self):
        return self.person.in_region_time > 2.7

    def frame_output(self):
        return (
            self.person.track_id,
            # self.person.bottom_center,
            self.person.tlbr,
            self.person.in_region_time,
        )


def output_video():
    from vqpy.utils.visualize import save_output_video
    import os

    def vobj_annotations(filtered_vobj):
        track_id = filtered_vobj["track_id"]
        time = filtered_vobj["in_region_time"]

        return f"Time: {time}, Id: {track_id}"

    def global_annotations(line):
        filtered_vobjs = line["Person"]
        all_in_region_time = [
            vobj["in_region_time"] for vobj in filtered_vobjs
        ]
        if len(all_in_region_time) == 0:
            return ["No person in queue"]
        number_of_people = len(all_in_region_time)
        average_time = round(
            sum(all_in_region_time) / len(all_in_region_time), 2
        )
        max_time = max(all_in_region_time)
        min_time = min(all_in_region_time)
        return [
            f"Number of People: {number_of_people}",
            f"Average Time: {average_time}",
            f"Max Time: {max_time}",
            f"Min Time: {min_time}",
        ]

    save_output_video(
        video_path=args.path,
        query_result_path=os.path.join(
            args.save_folder, "QueueAnalysis_20230912_024152.json"
        ),
        query_class_name="Person",
        output_video_path=os.path.join(args.save_folder, "output.mp4"),
        sample_images=[100],
        regions=REGIONS,
        vobj_annotations=vobj_annotations,
        global_annotations=global_annotations,
    )


def output_video_track_id_only():
    import os
    from vqpy.utils.visualize import save_output_video

    save_output_video(
        video_path=args.path,
        query_result_path=os.path.join(
            args.save_folder, "QueueAnalysis_20230911_170859.json"
        ),
        query_class_name="Person",
        output_video_path=os.path.join(
            args.save_folder, "output-track_id.mp4"
        ),
    )


if __name__ == "__main__":
    args = make_parser().parse_args()
    REGIONS = [ast.literal_eval(args.polygon)]

    query_executor = vqpy.init(
        video_path=args.path,
        query_obj=QueueAnalysis(),
        verbose=True,
        output_per_frame_results=True,
    )
    import time
    st = time.time()
    vqpy.run(query_executor, save_folder=args.save_folder)
    print(f"total time: {time.time() - st}")

    # output_video()
    # output_video_track_id_only()
