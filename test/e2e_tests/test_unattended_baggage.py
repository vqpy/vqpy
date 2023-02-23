from pathlib import Path

import vqpy
from vqpy.class_names.coco import COCO_CLASSES
from vqpy.operator.detector import register

from e2e import FakeYOLOX, compare

root = Path(__file__).parent.parent.parent
video_name = "unattended_baggage"
video_ext = "mp4"
task_name = "unattended_baggage"
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


def test_unattended_baggage(setup_example_path):
    from unattended_baggage.main import (
        Person,
        Baggage,
        FindUnattendedBaggage,
    )  # noqa: E402

    register(fake_detector_name, FakeYOLOX, precomputed_path, None)
    vqpy.launch(
        cls_name=COCO_CLASSES,
        cls_type={"person": Person, "backpack": Baggage, "suitcase": Baggage},
        tasks=[FindUnattendedBaggage()],
        video_path=video_path,
        save_folder=save_folder,
        detector_name=fake_detector_name,
    )

    compare(result_path, expected_result_path)
