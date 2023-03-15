from vqpy.backend.plan import Planner, Executor
from vqpy.frontend.vobj import VObjBase, vobj_property
from vqpy.frontend.query import QueryBase
import os
import fake_yolox # noqa F401
import numpy as np
import math


current_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(current_dir, "..", "..", "resources/")
video_path = os.path.join(resource_dir, "pedestrian_10s.mp4")


class Vehicle(VObjBase):

    def __init__(self) -> None:
        self.class_name = "car"
        self.object_detector = "fake_yolox"
        self.detector_kwargs = {"device": "cpu"}
        super().__init__()

    @vobj_property(inputs={"tlbr": 0})
    def license_plate(self, values):
        tlbr = values["tlbr"]
        assert isinstance(tlbr, np.ndarray)
        assert len(tlbr) == 4
        if tlbr[1] < 100:
            return "ABC123"
        else:
            return "XYZ789"

    @vobj_property(inputs={"tlbr": 1})
    def velocity(self, values):
        fps = 24.0
        last_tlbr, tlbr = values["tlbr"]
        if last_tlbr is None or tlbr is None:
            return 0
        last_center = (last_tlbr[:2] + last_tlbr[2:]) / 2
        cur_center = (tlbr[:2] + tlbr[2:]) / 2
        tlbr_avg = (tlbr + last_tlbr) / 2
        scale = (tlbr_avg[3] - tlbr_avg[1]) / 1.5
        dcenter = (cur_center - last_center) / scale * fps
        return math.sqrt(sum(dcenter * dcenter))


class ListVehicle(QueryBase):

    def __init__(self) -> None:
        self.car = Vehicle()

    def frame_constraint(self):
        return (self.car.score > 0.6) & (self.car.score < 0.7) & \
            (self.car.license_plate.cmp(lambda x: x.startswith("ABC")))

    def frame_output(self):
        return self.car.license_plate


def test_plan():

    planner = Planner()
    launch_args = {
        "video_path": video_path,
    }
    root_plan_node = planner.parse(ListVehicle())
    planner.print_plan(root_plan_node)
    executor = Executor(root_plan_node, launch_args)
    result = executor.execute()

    for frame in result:
        # todo: put in planner
        print(frame.id)
        for person_idx in frame.filtered_vobjs[0]["car"]:
            print(frame.vobj_data["car"][person_idx])


if __name__ == "__main__":
    test_plan()
