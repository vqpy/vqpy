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
sys.path.append(fall_detection_lib_path)


class Person(vqpy.VObjBase):
    required_fields = ["class_id", "tlbr"]

    # default values, to be assigned in main()
    pose_model = None
    action_model = None

    @vqpy.property()
    @vqpy.stateful(30)
    def keypoints(self):
        # per-frame property, but tracker can return objects
        # not in the current frame
        image = self._ctx.frame
        tlbr = self.getv("tlbr")
        if tlbr is None:
            return None
        return Person.pose_model.predict(image, torch.tensor([tlbr]))

    @vqpy.property()
    def pose(self) -> str:
        keypoints_list = []
        for i in range(-self._track_length, 0):
            keypoint = self.getv("keypoints", i)
            if keypoint is not None:
                keypoints_list.append(keypoint)
            if len(keypoints_list) >= 30:
                break
        if len(keypoints_list) < 30:
            return "unknown"
        pts = np.array(keypoints_list, dtype=np.float32)
        out = Person.action_model.predict(pts, self._ctx.frame.shape[:2])
        action_name = Person.action_model.class_names[out[0].argmax()]
        return action_name


class FallDetection(vqpy.QueryBase):
    """The class obtaining all fallen person"""

    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        filter_cons = {
            "__class__": lambda x: x == Person,
            "pose": lambda x: x == "Fall Down",
        }
        select_cons = {"track_id": None, "tlbr": lambda x: str(x)}
        return vqpy.VObjConstraint(
            filter_cons=filter_cons, select_cons=select_cons, filename="fall"
        )


def test_fall_detection():
    # avoid randomness
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
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
