import argparse
import vqpy
from yolox_detector import YOLOXDetector
from vqpy.detector.logger import register


def make_parser():
    parser = argparse.ArgumentParser("VQPy Demo!")
    parser.add_argument("--path", help="path to video")
    parser.add_argument(
        "--save_folder",
        default=None,
        help="the folder to save the final result",
    )
    parser.add_argument(
        "-d", "--pretrained_model_dir", help="Directory to pretrained models"
    )
    return parser


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
    @vqpy.property()
    @vqpy.cross_vobj_property(
        vobj_type=Person, vobj_num="ALL",
        vobj_input_fields=("track_id", "tlbr")
    )
    # function decorator responsible for retrieving list of properties
    # Person_id and Person_tlbr given as a list of track_id's and tlbr's
    def owner(self, person_ids, person_tlbrs):
        # if previous owner within distance, return previous owner track id
        # else: find the nearest person within distance, return the track id
        # else: return None
        baggage_tlbr = self.getv('tlbr')
        prev_owner = self.getv('owner', -2)
        owner_id = None

        # return previous owner, if baggage is not present in current frame
        # return None is previous owner is not present in current frame
        if baggage_tlbr is None:
            return prev_owner if prev_owner in person_ids else None

        # set threshold to baggage's width
        threshold = (baggage_tlbr[3] - baggage_tlbr[1])
        min_dist = threshold + 1
        for person_id, person_tlbr in zip(person_ids, person_tlbrs):
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
            "owner": vqpy.utils.continuing(
                condition=lambda x: x is None, duration=3, name="no_owner"
            ),
        }
        select_cons = {
            "track_id": None,
            "tlbr": lambda x: str(x),
            "no_owner_periods": lambda x: str(x),
        }
        return vqpy.VObjConstraint(
            filter_cons=filter_cons,
            select_cons=select_cons,
            filename="unattended_baggage",
        )


if __name__ == "__main__":
    args = make_parser().parse_args()
    register("yolox", YOLOXDetector, "yolox_x.pth")
    vqpy.launch(
        cls_name=vqpy.COCO_CLASSES,
        cls_type={"person": Person, "backpack": Baggage, "suitcase": Baggage},
        tasks=[FindUnattendedBaggage()],
        video_path=args.path,
        save_folder=args.save_folder,
        detector_name="yolox",
        detector_model_dir=args.pretrained_model_dir,
    )
