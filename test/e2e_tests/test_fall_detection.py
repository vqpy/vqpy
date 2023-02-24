import pytest
from pathlib import Path
import torch
import numpy as np
import random
import sys

import vqpy
from vqpy.operator.detector import register

from e2e import FakeYOLOX, compare

root = Path(__file__).parent.parent.parent
video_name = "fall"
video_ext = "mp4"
task_name = "fall"
fake_detector_name = f"fake-yolox-{video_name}"
video_path = (root / f"videos/{video_name}.{video_ext}").as_posix()
precomputed_path = (root / f"precomputed/{video_name}_yolox.pkl").as_posix()
save_folder = (root / "e2e_outputs/").as_posix()
result_path = (
    root / f"e2e_outputs/{video_name}_{task_name}_{fake_detector_name}.json"
).as_posix()
expected_result_path = (
    root / f"expected_results/{video_name}_{task_name}_yolox.json"
).as_posix()

fall_detection_lib_path = (root / "Human-Falling-Detect-Tracks").as_posix()


@pytest.fixture
def setup_fall_detection_lib_path():
    sys.path.append(fall_detection_lib_path)
    yield
    sys.path.remove(fall_detection_lib_path)


def test_fall_detection(setup_example_path, setup_fall_detection_lib_path):
    # avoid randomness
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    from fall_detection.main import Person, FallDetection  # noqa: E402
    from PoseEstimateLoader import SPPE_FastPose  # noqa: E402
    from ActionsEstLoader import TSSTG  # noqa: E402

    register(fake_detector_name, FakeYOLOX, precomputed_path, None)
    pose_model = SPPE_FastPose(
        backbone="resnet50",
        device="cpu",
        weights_file=(root / "models/fast_res50_256x192.pth").as_posix(),
    )
    action_model = TSSTG((root / "models/tsstg-model.pth").as_posix())
    Person.pose_model = pose_model
    Person.action_model = action_model
    vqpy.launch(
        cls_name=vqpy.COCO_CLASSES,
        cls_type={"person": Person},
        tasks=[FallDetection()],
        video_path=video_path,
        save_folder=save_folder,
        detector_name=fake_detector_name,
    )

    compare(result_path, expected_result_path)
