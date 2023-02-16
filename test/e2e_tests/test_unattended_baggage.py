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


def distance(vobj1_tlbr, vobj2_tlbr):
    from math import sqrt
    center1 = (vobj1_tlbr[:2] + vobj1_tlbr[2:]) / 2
    center2 = (vobj2_tlbr[:2] + vobj2_tlbr[2:]) / 2
    # difference between center coordinate
    diff = center2 - center1
    return sqrt(diff[0]**2 + diff[1]**2)


class Person(vqpy.VObjBase):
    pass


class Baggage(vqpy.VObjBase):
    @vqpy.stateful(length=2)
    @vqpy.cross_vobj_property(
        vobj_type=Person, vobj_num="ALL",
        vobj_input_fields=("track_id", "tlbr")
    )
    # function decorator responsible for retrieving list of properties
    # Person_id and Person_tlbr given as a list of track_id's and tlbr's
    def owner(self, person_ids_tlbrs):
        # if previous owner within distance, return previous owner track id
        # else: find the nearest person within distance, return the track id
        # else: return None
        baggage_tlbr = self.getv('tlbr')
        prev_owner = self.getv('owner', -2)
        owner_id = None
        # with new implementation of @cross_vobj_property and VObj.update, only
        # tracked VObjs will go into the filter and compute property "owner"
        # Thus no need to handle situation of baggage_tlbr being None

        # set threshold to baggage's width
        threshold = (baggage_tlbr[3] - baggage_tlbr[1])
        min_dist = threshold + 1
        for person_id, person_tlbr in person_ids_tlbrs:
            dist = distance(baggage_tlbr, person_tlbr)
            if person_id == prev_owner and dist <= threshold:
                # return previous owner if still around
                return prev_owner
            if dist <= threshold and dist < min_dist:
                owner_id = person_id
                min_dist = dist
        # new owner is returned (will return None if owner not found)
        return owner_id


class FindUnattendedBaggage(vqpy.QueryBase):
    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        filter_cons = {
            "__class__": lambda x: x == Baggage,
            "owner": vqpy.query.continuing(
                condition=lambda x: x is None, duration=10, name="no_owner"
            ),
        }
        select_cons = {
            "track_id": None,
            "tlbr": lambda x: str(x),
            "no_owner_periods": None,
        }
        return vqpy.VObjConstraint(
            filter_cons=filter_cons,
            select_cons=select_cons,
            filename="unattended_baggage",
        )


def test_unattended_baggage():
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
