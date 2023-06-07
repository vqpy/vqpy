from vqpy.backend.operator.vobj_projector import VObjProjector
from vqpy.backend.operator.object_detector import ObjectDetector
from vqpy.backend.operator.video_reader import VideoReader
from vqpy.backend.operator.vobj_filter import VObjFilter
from vqpy.backend.operator.tracker import Tracker
from vqpy.common import InvalidProperty
from collections import defaultdict


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


@pytest.fixture
def tracker(video_reader, object_detector):
    fps = video_reader.metadata["fps"]
    tracker = Tracker(
        prev=object_detector,
        tracker_name="byte",
        class_name="person",
        fps=fps,
    )
    return tracker


@pytest.fixture
def stateless_filter(object_detector):
    person_vobj_filter = VObjFilter(
        prev=object_detector,
        condition_func="person",
    )
    return person_vobj_filter


@pytest.fixture
def stateful_filter(tracker):
    person_vobj_filter = VObjFilter(
        prev=tracker,
        condition_func="person",
    )
    return person_vobj_filter


def dep_tlbr_score(values):
    tlbr_score = values["tlbr_score"]
    return tlbr_score


def test_stateless_projector(stateless_filter):
    # test projector with no history
    def tlbr_score(values):
        tlbr = values["tlbr"]  # noqa: F841
        score = values["score"]
        return score > 0.5

    projector = VObjProjector(
        prev=stateless_filter,
        property_name="score_gt_0.5",
        property_func=tlbr_score,
        dependencies={"tlbr": 0, "score": 0},
        class_name="person",
    )
    while projector.has_next():
        frame = projector.next()
        for vobj in frame.vobj_data["person"]:
            data = {"tlbr": vobj["tlbr"], "score": vobj["score"]}
            assert vobj["score_gt_0.5"] == tlbr_score(data)
    assert projector._hist_buffer.empty


def test_projector_dep_non_computed(stateless_filter):
    projector = VObjProjector(
        prev=stateless_filter,
        property_name="dep_tlbr_score",
        property_func=dep_tlbr_score,
        dependencies={"tlbr_score": 0},
        class_name="person",
    )

    with pytest.raises(AssertionError):
        if projector.has_next():
            projector.next()


def test_projector_non_filter_index(stateless_filter):

    projector = VObjProjector(
        prev=stateless_filter,
        property_name="dep_tlbr_score",
        property_func=dep_tlbr_score,
        dependencies={"tlbr_score": 0},
        class_name="person",
        filter_index=1,
    )

    with pytest.raises(ValueError):
        if projector.has_next():
            projector.next()


def test_stateful_projector(stateful_filter):
    # test projector with history

    def hist_2_scores(values):
        last_2_score = values["score"]
        assert len(last_2_score) == 2
        if last_2_score[0] is None or last_2_score[1] is None:
            return 0
        return last_2_score

    hist_len = 1

    projector = VObjProjector(
        prev=stateful_filter,
        property_name="hist_scores",
        property_func=hist_2_scores,
        dependencies={"score": hist_len},
        class_name="person",
    )
    checked = False
    result = defaultdict(list)

    while projector.has_next():
        frame = projector.next()
        for vobj in frame.vobj_data["person"]:
            track_id = vobj.get("track_id")
            if track_id:
                checked = True
                if frame.id < hist_len:
                    assert isinstance(vobj["hist_scores"], InvalidProperty)
                else:
                    assert not isinstance(vobj["hist_scores"], InvalidProperty)
                    hist_buffer = projector._hist_buffer
                    row = (hist_buffer["track_id"] == track_id) & \
                        (hist_buffer["frame_id"] == frame.id)
                    assert hist_buffer.loc[row, "score"].values[0] == \
                        vobj["score"]
                    data = {"frame_id": frame.id,
                            "hist_scores": vobj["hist_scores"]}
                    result[track_id].append(data)

    for vobj_data in result.items():
        last_data = None
        for data in vobj_data:
            if last_data is not None and \
                    last_data["frame_id"] + 1 == data["frame_id"]:
                assert data["hist_scores"][0] == last_data["hist_scores"][1]
                last_data = data

    assert not projector._hist_buffer.empty
    assert checked

    # test delete history


def test_stateless_projector_image_video(stateless_filter):
    # property function depend on image and video metadata
    def image_fps_stateless(values):
        image = values["image"]
        fps = values["fps"]
        assert fps == 24.0
        assert image.shape[2] == 3
        return 1

    projector = VObjProjector(
        prev=stateless_filter,
        property_name="image_fps_stateless",
        property_func=image_fps_stateless,
        dependencies={"image": 0, "fps": 0},
        class_name="person",
    )
    num_image_cropped = 0
    while projector.has_next():
        frame = projector.next()
        for vobj_data in frame.vobj_data["person"]:
            assert "image" not in vobj_data
            assert "fps" not in vobj_data
            result = vobj_data["image_fps_stateless"]
            if result == 1:
                num_image_cropped += 1
    assert projector._hist_buffer.empty
    assert num_image_cropped > 0


def test_stateful_projector_image_video(stateful_filter):
    # property function depend on image and video metadata
    def image_fps_stateful(values):
        last_2_images = values["image"]
        assert len(last_2_images) == 2

        fps = values["fps"]
        assert fps == 24.0
        for image in last_2_images:
            if image is None:
                return 0
            else:
                assert image.shape[2] == 3
        return 1

    hist_len = 1

    projector = VObjProjector(
        prev=stateful_filter,
        property_name="image_fps_stateful",
        property_func=image_fps_stateful,
        dependencies={"image": hist_len, "fps": 0},
        class_name="person",
    )
    num_image_cropped = 0
    while projector.has_next():
        frame = projector.next()

        for vobj in frame.vobj_data["person"]:
            track_id = vobj.get("track_id")
            if track_id:
                if frame.id < hist_len:
                    assert isinstance(vobj["image_fps_stateful"],
                                      InvalidProperty)
                else:
                    result = vobj["image_fps_stateful"]
                    if result == 1:
                        num_image_cropped += 1
    assert not projector._hist_buffer.empty
    assert num_image_cropped > 0


def test_stateful_projector_dep_self(stateful_filter):

    def self_dep(values):
        # only depend on last self value
        last2_value, last_value, this_value = values["self_dep"]
        assert this_value is None
        if last_value is not None:
            return last_value + 1
        else:
            return 0

    hist_len = 2
    projector = VObjProjector(
        prev=stateful_filter,
        property_name="self_dep",
        property_func=self_dep,
        dependencies={"self_dep": hist_len},
        class_name="person",
    )
    checked = False
    result = defaultdict(list)

    while projector.has_next():
        frame = projector.next()
        for vobj in frame.vobj_data["person"]:
            track_id = vobj.get("track_id")
            if track_id:
                if frame.id < hist_len:
                    assert vobj["self_dep"] is None
                else:
                    data = {"frame_id": frame.id, "self_dep": vobj["self_dep"]}
                    result[track_id].append(data)
                    assert vobj["self_dep"] is not None
    for vobj_data in result.values():
        last_data = None
        for data in vobj_data:
            if last_data is not None and \
                    last_data["frame_id"] + 1 == data["frame_id"]:
                assert data["self_dep"] == last_data["self_dep"] + 1
                checked = True
            last_data = data
            assert not projector._hist_buffer.empty
    assert checked


# def test_stateful_projector_multi_deps():

#     def hist_tlbr_score(values):
#         last_5_tlbrs = values["last_5_tlbrs"]
#         cur_score = values["cur_score"]
#         assert len(last_5_tlbrs) == 5
#         return cur_score - 0.5


def test_projector_multi_index():
    pass


def test_graph():
    pass
