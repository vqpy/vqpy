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


class Person(VObjBase):

    def __init__(self) -> None:
        self.class_name = "person"
        self.object_detector = "fake_yolox"
        self.detector_kwargs = {"device": "cpu"}
        super().__init__()

    @vobj_property(inputs={"tlbr": 0})
    def center(self, values):
        tlbr = values["tlbr"]
        assert isinstance(tlbr, np.ndarray)
        assert len(tlbr) == 4
        return (tlbr[:2] + tlbr[2:]) / 2

    @vobj_property(inputs={"tlbr": 2})
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

    @vobj_property(inputs={"velocity": 2})
    def acceleration(self, values):
        fps = 24.0
        last_velocity, velocity = values["velocity"]
        if last_velocity is None or velocity is None:
            return 0
        return (velocity - last_velocity) * fps


class ListPerson(QueryBase):

    def __init__(self) -> None:
        self.person = Person()

    def frame_constraint(self):
        return (self.person.score > 0.6) & (self.person.score < 0.7) & \
            (self.person.velocity > 0) & (self.person.acceleration > 0)

    def frame_output(self):
        return {
            "center": self.person.center,
            "velocity": self.person.velocity,
            "acceleration": self.person.acceleration,
        }


def test_plan():

    planner = Planner()
    launch_args = {
        "video_path": video_path,
    }
    root_plan_node = planner.parse(ListPerson())
    planner.print_plan(root_plan_node)
    executor = Executor(root_plan_node, launch_args)
    result = executor.execute()

    for frame in result:
        # todo: put in planner
        print(frame.id)
        for person_idx in frame.filtered_vobjs[0]["person"]:
            print(frame.vobj_data["person"][person_idx])


if __name__ == "__main__":
    test_plan()
