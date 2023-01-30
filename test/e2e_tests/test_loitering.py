from pathlib import Path

import vqpy
from vqpy.class_names.coco import COCO_CLASSES
from vqpy.operator.detector import register

from e2e import FakeYOLOX, compare

root = Path(__file__).parent.parent.parent.parent
video_path = (root / "videos/loitering.mp4").as_posix()
precomputed_path = (root / "precomputed/loitering_yolox.pkl").as_posix()
save_folder = (root / "e2e_outputs/").as_posix()
result_path = (root / "e2e_outputs/loitering_loitering_fake-yolox.json").as_posix()
expected_path = (root / "expected/loitering_loitering_yolox.json").as_posix()


class Person(vqpy.VObjBase):
    pass


class People_loitering_query(vqpy.QueryBase):
    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        REGION = [(550, 550), (1162, 400), (1720, 720), (1430, 1072), (600, 1073)]
        REGIONS = [REGION]

        filter_cons = {
            "__class__": lambda x: x == Person,
            "bottom_center": vqpy.query.continuing(
                condition=vqpy.query.utils.within_regions(REGIONS),
                duration=10,
                name="in_roi",
            ),
        }
        select_cons = {
            "track_id": None,
            "coordinate": lambda x: str(x),
            "in_roi_periods": None,
        }
        return vqpy.VObjConstraint(filter_cons, select_cons, filename="loitering")


def test_loitering():
    register("fake-yolox", FakeYOLOX, precomputed_path, None)
    vqpy.launch(
        cls_name=COCO_CLASSES,
        cls_type={"person": Person},
        tasks=[People_loitering_query()],
        video_path=video_path,
        save_folder=save_folder,
        detector_name="fake-yolox",
    )

    compare(result_path, expected_path)
