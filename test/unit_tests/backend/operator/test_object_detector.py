from vqpy.backend.operator.object_detector import ObjectDetector
from vqpy.backend.operator.video_reader import VideoReader

import pytest
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(current_dir, "..", "..", "resources/")


@pytest.fixture()
def video_reader():
    video_path = os.path.join(resource_dir, "pedestrian_10s.mp4")
    assert os.path.isfile(video_path)
    video_reader = VideoReader(video_path)
    return video_reader


def test_single_class_object_detector(video_reader):
    object_detector = ObjectDetector(
        prev=video_reader,
        class_names="person",
        detector_name="yolox",
    )

    person_detected = False
    while object_detector.has_next():
        frame = object_detector.next()
        if frame.vobj_data:
            assert frame.vobj_data.keys() == {"person"}
            person_0 = frame.vobj_data["person"][0]
            assert person_0["tlbr"].shape == (4,)
            assert 1 > person_0["score"] > 0 
            person_detected = True
    assert person_detected


def test_multi_class_object_detector(video_reader):
    object_detector = ObjectDetector(
        prev=video_reader,
        class_names={"car", "truck"},
        detector_name="yolox",
    )
    car_detected = False
    while object_detector.has_next():
        frame = object_detector.next()
        if frame.vobj_data:
            if "car" in frame.vobj_data:
                car_0 = frame.vobj_data["car"][0]
                assert car_0["tlbr"].shape == (4,)
                assert 1 > car_0["score"] > 0
                car_detected = True
            else:
                assert "truck" in frame.vobj_data
    assert car_detected
