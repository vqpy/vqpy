from vqpy.operator.video_reader import VideoReader
from vqpy.obj.frame_new import Frame
import pytest
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
resource_dir = os.path.join(current_dir, "..", "resources/")


@pytest.fixture(scope='module')
def video_reader():
    video_path = os.path.join(resource_dir, "pedestrian_10s.mp4")
    assert os.path.isfile(video_path)
    video_reader = VideoReader(video_path)
    yield video_reader
    del video_reader


def test_video_metadata(video_reader):
    metadata = video_reader.metadata
    assert metadata["frame_width"]
    assert metadata["frame_height"]
    assert metadata["fps"]
    assert metadata["n_frames"]


def test_next(video_reader):
    counter = 0
    while video_reader.has_next():
        frame = video_reader.next()
        assert isinstance(frame, Frame)
        assert frame.frame_id == counter
        assert frame.video_metadata == video_reader.metadata
        frame_height = frame.video_metadata["frame_height"]
        frame_width = frame.video_metadata["frame_width"]
        assert frame.frame_image.shape == (frame_height, frame_width, 3)
        counter += 1

