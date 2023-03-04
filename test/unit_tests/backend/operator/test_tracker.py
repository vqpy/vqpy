from vqpy.backend.operator.object_detector import ObjectDetector
from vqpy.backend.operator.video_reader import VideoReader
from vqpy.backend.operator.vobj_filter import VObjFilter
from vqpy.backend.operator.tracker import Tracker

import pytest
import os
import fake_yolox  # noqa: F401
current_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(current_dir, "..", "..", "resources/")


@pytest.fixture
def video_reader():
    video_path = os.path.join(resource_dir, "pedestrian_10s.mp4")
    assert os.path.isfile(video_path)
    video_reader = VideoReader(video_path)
    return video_reader


@pytest.fixture
def object_detector(video_reader):
    object_detector = ObjectDetector(
        prev=video_reader,
        class_names={"person", "car", "truck"},
        detector_name="fake_yolox",
        detector_kwargs={"device": "cpu"}
    )
    return object_detector


def test_tracker_1(object_detector, video_reader):
    fps = video_reader.metadata["fps"]
    tracker = Tracker(
        prev=object_detector,
        tracker_name="byte",
        class_name="person",
        fps=fps,
    )
    counter = 0
    while tracker.has_next():
        frame = tracker.next()
        if "person" in frame.vobj_data:
            num_person = len(frame.vobj_data["person"])
            num_person_tracked = len([p for p in frame.vobj_data["person"]
                                      if p.get("track_id")])
            assert 0 < num_person_tracked <= num_person
        counter += 1
    assert counter == video_reader.metadata["n_frames"]


def test_tracker_2(object_detector, video_reader):
    fps = video_reader.metadata["fps"]
    person_tracker = Tracker(
        prev=object_detector,
        class_name="person",
        fps=fps,
    )
    car_tracker = Tracker(
        prev=person_tracker,
        class_name="car",
        fps=fps,
    )
    counter = 0
    total_num_car_tracked = 0
    while car_tracker.has_next():
        frame = car_tracker.next()
        if "person" in frame.vobj_data:
            num_person = len(frame.vobj_data["person"])
            num_person_tracked = len([p for p in frame.vobj_data["person"]
                                      if p.get("track_id")])
            assert 0 < num_person_tracked <= num_person
        if "car" in frame.vobj_data:
            num_car_tracked = len([c for c in frame.vobj_data["car"]
                                   if c.get("track_id")])
            total_num_car_tracked += num_car_tracked
        counter += 1
    assert counter == video_reader.metadata["n_frames"]
    assert total_num_car_tracked > 0


def test_filter_index_1(object_detector, video_reader):
    fps = video_reader.metadata["fps"]
    class_name = "person"

    person_vobj_filter = VObjFilter(
        prev=object_detector,
        condition_func=class_name,
        filter_index=0,
    )
    tracker = Tracker(
        prev=person_vobj_filter,
        class_name=class_name,
        filter_index=0,
        fps=fps,
    )
    counter = 0
    while tracker.has_next():
        frame = tracker.next()
        if class_name in frame.vobj_data:
            num_person = len(frame.vobj_data[class_name])
            num_person_tracked = len([p for p in frame.vobj_data[class_name]
                                      if p.get("track_id")])
            assert 0 < num_person_tracked <= num_person
        counter += 1
    assert counter == video_reader.metadata["n_frames"]


def test_filter_index_2(object_detector, video_reader):
    fps = video_reader.metadata["fps"]

    person_vobj_filter = VObjFilter(
        prev=object_detector,
        condition_func="person",
        filter_index=0,
    )
    person_tracker = Tracker(
        prev=person_vobj_filter,
        class_name="person",
        filter_index=0,
        fps=fps,
    )
    car_vobj_filter = VObjFilter(
        prev=person_tracker,
        condition_func="car",
        filter_index=1,
    )
    car_tracker = Tracker(
        prev=car_vobj_filter,
        class_name="car",
        filter_index=1,
        fps=fps,
    )

    counter = 0
    total_num_car_tracked = 0
    while car_tracker.has_next():
        frame = car_tracker.next()
        if "person" in frame.vobj_data:
            num_person = len(frame.vobj_data["person"])
            num_person_tracked = len([p for p in frame.vobj_data["person"]
                                      if p.get("track_id")])
            assert 0 < num_person_tracked <= num_person
        if "car" in frame.vobj_data:
            num_car_tracked = len([c for c in frame.vobj_data["car"]
                                   if c.get("track_id")])
            total_num_car_tracked += num_car_tracked
        counter += 1
    assert counter == video_reader.metadata["n_frames"]
    assert total_num_car_tracked > 0
