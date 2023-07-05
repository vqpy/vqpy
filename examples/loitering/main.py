import argparse
import vqpy


def make_parser():
    parser = argparse.ArgumentParser("VQPy Demo!")
    parser.add_argument("--path", help="path to video")
    parser.add_argument(
        "--save_folder",
        default=None,
        help="the folder to save the final result",
    )
    return parser


class Person(vqpy.VObjBase):
    pass


class People_loitering_query(vqpy.QueryBase):
    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        REGION = [
            (550, 550),
            (1162, 400),
            (1720, 720),
            (1430, 1072),
            (600, 1073),
        ]
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
            "coordinate": lambda x: str(x),  # convert to string for
            # JSON serialization
            # name in vqpy.continuing + '_periods' can be used in select_cons.
            "in_roi_periods": None,
        }
        return vqpy.VObjConstraint(
            filter_cons, select_cons, filename="loitering"
        )


if __name__ == "__main__":
    args = make_parser().parse_args()
    vqpy.launch(
        cls_name=vqpy.COCO_CLASSES,
        cls_type={"person": Person},
        tasks=[People_loitering_query()],
        video_path=args.path,
        save_folder=args.save_folder,
    )
