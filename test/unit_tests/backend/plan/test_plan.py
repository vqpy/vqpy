from vqpy.backend.plan import Planer, Executor
from vqpy.frontend.vobj import VObjBase, stateful
from vqpy.frontend.query import QueryBase
import pytest
import os
import fake_yolox
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(current_dir, "..", "..", "resources/")
video_path = os.path.join(resource_dir, "pedestrian_10s.mp4")


class Vehicle(VObjBase):

    def __init__(self) -> None:
        self.class_name = "car"
        self.object_detector = "fake_yolox"
        self.detector_kwargs = {"device": "cpu"}
        super().__init__()

    @stateful(inputs={"tlbr", 0})
    def license_plate(self, tlbr):
        assert isinstance(tlbr, np.ndarray)
        assert len(tlbr) == 4
        return "ABC123"


class ListVehicle(QueryBase):

    def __init__(self) -> None:
        self.car = Vehicle()
        super().__init__()

    def frame_constraint(self):
        return (self.car.score > 0.6) & (self.car.score < 0.7)

    def frame_output(self):
        return self.car.license_plate


def test_plan():

    planer = Planer()
    launch_args = {
        "video_path": video_path,
    }
    root_plan_node = planer.parse(ListVehicle())
    planer.print_plan(root_plan_node)
    executor = Executor(root_plan_node, launch_args)
    result = executor.execute()

    for frame in result:
        print(frame)

if __name__ == "__main__":
    test_plan()