from vqpy.backend.operator.vobj_projector import VObjProjector
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


def tlbr_score(values):
    tlbr = values["tlbr"]  # noqa: F841
    score = values["score"]
    return score > 0.5


def dep_tlbr_score(values):
    tlbr_score = values["tlbr_score"]
    return tlbr_score


def hist_tlbr(values):
    last_2_tlbrs = values["tlbr"]
    assert len(last_2_tlbrs) == 2
    for tlbr in last_2_tlbrs:
        if tlbr is None:
            return 0
    return last_2_tlbrs[0][0] - last_2_tlbrs[1][0]


def hist_tlbr_score(values):
    last_5_tlbrs = values["last_5_tlbrs"]
    cur_score = values["cur_score"]
    assert len(last_5_tlbrs) == 5
    return cur_score - 0.5


def dep_self(values):
    last_5_dep_selfs = values["dep_self"]
    assert len(last_5_dep_selfs) == 5
    assert last_5_dep_selfs[-1] is None
    assert all([dep_self == 1 for dep_self in last_5_dep_selfs[:-1]])
    return 1


def test_stateless_projector(object_detector):
    # test projector with no history
    person_vobj_filter = VObjFilter(
        prev=object_detector,
        condition_func="person",
    )

    projector = VObjProjector(
        prev=person_vobj_filter,
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


def test_projector_dep_non_computed(object_detector):
    person_vobj_filter = VObjFilter(
        prev=object_detector,
        condition_func="person",
    )
    projector = VObjProjector(
        prev=person_vobj_filter,
        property_name="dep_tlbr_score",
        property_func=dep_tlbr_score,
        dependencies={"tlbr_score": 0},
        class_name="person",
    )

    with pytest.raises(AssertionError):
        if projector.has_next():
            projector.next()


def test_projector_non_filter_index(object_detector):
    person_vobj_filter = VObjFilter(
        prev=object_detector,
        condition_func="person",
    )
    projector = VObjProjector(
        prev=person_vobj_filter,
        property_name="dep_tlbr_score",
        property_func=dep_tlbr_score,
        dependencies={"tlbr_score": 0},
        class_name="person",
        filter_index=1,
    )

    with pytest.raises(ValueError):
        if projector.has_next():
            projector.next()


def test_stateful_projector(tracker):
    # test projector with history
    person_vobj_filter = VObjFilter(
        prev=tracker,
        condition_func="person",
    )
    hist_len = 2

    projector = VObjProjector(
        prev=person_vobj_filter,
        property_name="hist_tlbr",
        property_func=hist_tlbr,
        dependencies={"tlbr": hist_len},
        class_name="person",
    )
    checked = False
    while projector.has_next():
        frame = projector.next()
        for vobj in frame.vobj_data["person"]:
            track_id = vobj.get("track_id")
            if track_id:
                checked = True
                if frame.id < hist_len - 1:
                    assert vobj["hist_tlbr"] is None
                else:
                    assert vobj["hist_tlbr"] is not None
                    # hist_buffer = projector._hist_buffer
                    # row = (hist_buffer["track_id"] == track_id) & \
                    #     (hist_buffer["frame_id"] == frame.id)
                    # assert hist_buffer.loc[row, "tlbr"] == vobj["tlbr"]
    assert not projector._hist_buffer.empty
    assert checked

    # test delete history


def test_stateless_projector_image_video(object_detector):
    # property function depend on image and video metadata
    def image_fps_stateless(values):
        image = values["image"]
        fps = values["fps"]
        assert fps == 24.0
        if image is not None:
            assert image.shape[2] == 3
            return 1
        return 0

    person_vobj_filter = VObjFilter(
        prev=object_detector,
        condition_func="person",
    )

    projector = VObjProjector(
        prev=person_vobj_filter,
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


def test_stateful_projector_image_video(tracker):
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

    person_vobj_filter = VObjFilter(
        prev=tracker,
        condition_func="person",
    )
    hist_len = 2

    projector = VObjProjector(
        prev=person_vobj_filter,
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
                if frame.id < hist_len - 1:
                    assert vobj["image_fps_stateful"] is None
                else:
                    result = vobj["image_fps_stateful"]
                    if result == 1:
                        num_image_cropped += 1
    assert not projector._hist_buffer.empty
    assert num_image_cropped > 0


def test_stateful_projector_dep_self():
    pass


def test_stateful_projector_multi_deps():
    pass


def test_projector_multi_index():
    pass


def test_graph():
    pass

# TODO: test property with video metadata and frame image dependencies
