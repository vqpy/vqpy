from typing import Dict
from vqpy.frontend.vobj import VObjBase, vobj_property
from vqpy.frontend.query import QueryBase
import os
import fake_yolox  # noqa F401
import numpy as np
import math
import vqpy
import time


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

    @vobj_property(inputs={"velocity": 1})
    def acceleration(self, values):
        time.sleep(0.005)
        fps = 24.0
        last_velocity, velocity = values["velocity"]
        if last_velocity is None or velocity is None:
            return 0
        return (velocity - last_velocity) * fps

    @vobj_property(inputs={"over_speed": 3, "velocity": 0})
    def over_speed(self, values):
        over_speed_values = values["over_speed"][:-1]
        this_velocity = values["velocity"]
        if all([x is None for x in over_speed_values]) and this_velocity > 0.6:
            return True
        if this_velocity > 0.6 and any(over_speed_values):
            return True
        return False


class ListPerson(QueryBase):
    def __init__(self) -> None:
        self.person = Person()

    def frame_constraint(self):
        return (
            (self.person.score > 0.6)
            & (self.person.score < 0.7)
            & (self.person.velocity < 1.0)
            & (self.person.acceleration > 0)
        )

    def frame_output(self):
        return (
            self.person.center,
            self.person.velocity,
            self.person.acceleration,
            self.person.over_speed,
        )


def test_plan():
    # planner = Planner()
    # launch_args = {
    #     "video_path": video_path,
    # }
    # root_plan_node = planner.parse(ListPerson())
    # planner.print_plan(root_plan_node)
    # executor = Executor(root_plan_node, launch_args)
    # result = executor.execute()
    # for res in result:
    #     print(res)
    query_executor = vqpy.init(video_path=video_path,
                               query_obj=ListPerson(),
                               output_per_frame_results=False)
    vqpy.run(query_executor)


def test_customize_video_reader():
    from vqpy.backend.operator import CustomizedVideoReader
    import cv2

    class MyVideoReader(CustomizedVideoReader):
        def __init__(self, video_path: str) -> None:
            self.video_path = video_path
            self.frame_id = -1
            self._cap = cv2.VideoCapture(video_path)
            super().__init__()

        def get_metadata(self) -> Dict:
            return {
                "fps": 24,
            }

        def has_next(self) -> bool:
            return self.frame_id + 1 < 200

        def _next(self):
            def timestamp_id():
                import datetime
                return datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

            self.frame_id += 1
            image = self._cap.read()[1]
            return {
                "image": image,
                "frame_id": self.frame_id,
                "ref_id": timestamp_id(),
            }

    video_reader = MyVideoReader(video_path=video_path)
    executor = vqpy.init(
        ListPerson(), custom_video_reader=video_reader, verbose=False,
        additional_frame_fields=["ref_id"]
    )
    result = vqpy.run(executor)
    return result


if __name__ == "__main__":
    st = time.time()
    test_plan()
    print(f"test_plan takes {time.time() - st:.2f} seconds")
    # result = test_customize_video_reader()
