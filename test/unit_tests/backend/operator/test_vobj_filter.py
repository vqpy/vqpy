from vqpy.backend.operator.object_detector import ObjectDetector
from vqpy.backend.operator.video_reader import VideoReader
from vqpy.backend.operator.vobj_filter import VObjFilter

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


def test_non_exist_class_name(object_detector):
    class_name = "none"
    assert object_detector.next()

    vobj_filter_non_exist_class = VObjFilter(
        prev=object_detector,
        condition_func=class_name,
        filter_index=0,
    )
    frame = vobj_filter_non_exist_class.next()
    assert frame.filtered_vobjs[0][class_name] == []


def test_class_name_diff_index(object_detector):
    person_vobj_filter = VObjFilter(
        prev=object_detector,
        condition_func="person",
        filter_index=0,
    )

    car_vobj_filter = VObjFilter(
        prev=person_vobj_filter,
        condition_func="car",
        filter_index=1,
    )
    while car_vobj_filter.has_next():
        frame = car_vobj_filter.next()
        assert len(frame.filtered_vobjs) == 2
        assert "person" not in frame.filtered_vobjs[1]
        assert "car" not in frame.filtered_vobjs[0]
        if "person" in frame.vobj_data:
            num_person = len(frame.vobj_data["person"])
            assert len(frame.filtered_vobjs[0]["person"]) == num_person
        else:
            assert frame.filtered_vobjs[0]["person"] == []
        if "car" in frame.vobj_data:
            num_car = len(frame.vobj_data["car"])
            assert len(frame.filtered_vobjs[1]["car"]) == num_car
        else:
            assert frame.filtered_vobjs[1]["car"] == []


def test_class_name_same_index(object_detector):
    truck_vobj_filter = VObjFilter(
        prev=object_detector,
        condition_func="truck",
        filter_index=0,
    )

    car_vobj_filter = VObjFilter(
        prev=truck_vobj_filter,
        condition_func="car",
        filter_index=0,
    )

    while car_vobj_filter.has_next():
        frame = car_vobj_filter.next()
        assert len(frame.filtered_vobjs) == 1

        if "truck" in frame.vobj_data:
            num_truck = len(frame.vobj_data["truck"])
            assert len(frame.filtered_vobjs[0]["truck"]) == num_truck
        else:
            assert frame.filtered_vobjs[0]["truck"] == []
        if "car" in frame.vobj_data:
            num_car = len(frame.vobj_data["car"])
            assert len(frame.filtered_vobjs[0]["car"]) == num_car
        else:
            assert frame.filtered_vobjs[0]["car"] == []


def test_multi_class_names(object_detector):
    vobj_filter = VObjFilter(
        prev=object_detector,
        condition_func=["truck", "car"],
        filter_index=0,
    )

    while vobj_filter.has_next():
        frame = vobj_filter.next()
        assert len(frame.filtered_vobjs) == 1

        if "truck" in frame.vobj_data:
            num_truck = len(frame.vobj_data["truck"])
            assert len(frame.filtered_vobjs[0]["truck"]) == num_truck
        else:
            assert frame.filtered_vobjs[0]["truck"] == []
        if "car" in frame.vobj_data:
            num_car = len(frame.vobj_data["car"])
            assert len(frame.filtered_vobjs[0]["car"]) == num_car
        else:
            assert frame.filtered_vobjs[0]["car"] == []


def test_existed_class(object_detector):
    car_vobj_filter = VObjFilter(
        prev=object_detector,
        condition_func="car",
        filter_index=0,
    )
    dup_car_vobj_filter = VObjFilter(
        prev=car_vobj_filter,
        condition_func="car",
        filter_index=0,
    )
    with pytest.raises(ValueError):
        dup_car_vobj_filter.next()
