from vqpy.backend.operator.object_detector import ObjectDetector
from vqpy.backend.operator.video_reader import VideoReader
from vqpy.backend.operator.frame_filter import FrameFilter

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
        class_names={"car", "truck"},
        detector_name="fake_yolox",
        detector_kwargs={"device": "cpu"}
    )
    return object_detector


def test_frame_filter(object_detector, video_reader):
    frame_filter = FrameFilter(
        prev=object_detector,
        condition_func=lambda frame: bool(frame.vobj_data)
    )
    counter = 0
    while frame_filter.has_next():
        frame = frame_filter.next()
        assert frame.vobj_data
        assert "car" in frame.vobj_data or "truck" in frame.vobj_data
        counter += 1
    assert 0 < counter < video_reader.metadata["n_frames"]


def test_frame_range_filter(video_reader):
    from vqpy.backend.operator.frame_filter import FrameRangeFilter
    frame_range_filter = FrameRangeFilter(
        prev=video_reader,
        frame_id_range=(10, 20),
    )
    counter = 0
    while frame_range_filter.has_next():
        frame = frame_range_filter.next()
        assert 10 <= frame.id < 20
        counter += 1
    assert counter == 10
