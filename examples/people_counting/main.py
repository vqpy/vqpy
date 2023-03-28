import argparse

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


class Person(VObjBase):
    def __init__(self) -> None:
        self.class_name = "person"
        self.object_detector = "yolox"
        self.detector_kwargs = {"device": "gpu"}
        super().__init__()

    @vobj_property(inputs={"tlbr": 5 - 1})
    def direction_vector(self, values):
        tlbr_c, tlbr_p = values["tlbr"][-1], values["tlbr"][0]
        if tlbr_c is None or tlbr_p is None:
            return None
        center_c = (tlbr_c[:2] + tlbr_c[2:]) / 2
        center_p = (tlbr_p[:2] + tlbr_p[2:]) / 2
        diff = center_c - center_p
        return int(diff[0]), int(diff[1])

    @vobj_property(inputs={"tlbr": 5 - 1})
    def direction(self, values):
        def denoise(target, reference):
            THRESHOLD = 10
            if target != 0 and reference / target >= THRESHOLD:
                target = 0
            return target

        def get_name(value, pos_name, neg_name):
            if value > 0:
                result = pos_name
            elif value < 0:
                result = neg_name
            else:
                result = ""
            return result

        def get_center(tlbr):
            return (tlbr[:2] + tlbr[2:]) / 2

        def most_frequent(List):
            from collections import Counter

            occurence_count = Counter(List)
            return occurence_count.most_common(1)[0][0]

        tlbr_past = values["tlbr"]
        if any(value is None for value in tlbr_past):
            return None
        centers = list(map(get_center, tlbr_past))
        diffs = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]

        diff_xs = [denoise(diff[0], diff[1]) for diff in diffs]
        diff_ys = [denoise(diff[1], diff[0]) for diff in diffs]

        horizontal = most_frequent(
            [get_name(diff_x, "right", "left") for diff_x in diff_xs]
        )
        vertical = most_frequent(
            [get_name(diff_y, "bottom", "top") for diff_y in diff_ys]
        )
        direction = vertical + horizontal
        if direction == "":
            direction = None

        return direction


class CountPersonHeadLeft(QueryBase):
    def __init__(self):
        self.person = Person()

    def frame_constraint(self):
        return self.person.direction.cmp(
            lambda x: x is not None and "left" in x
        )

    def frame_output(self):
        return {
            "direction": self.person.direction,
            "direction_vector": self.person.direction_vector,
        }


if __name__ == "__main__":
    args = make_parser().parse_args()
    planner = Planner()
    launch_args = {"video_path": args.path}
    root_plan_node = planner.parse(CountPersonHeadLeft())
    planner.print_plan(root_plan_node)
    executor = Executor(root_plan_node, launch_args)
    result = executor.execute()

    for frame in result:
        print(frame.id)
        for person_idx in frame.filtered_vobjs[0]["person"]:
            print(frame.vobj_data["person"][person_idx])
