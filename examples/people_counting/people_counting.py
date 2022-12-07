import argparse
import vqpy
from yolox_detector import YOLOXDetector
from vqpy.detector.logger import register


def make_parser():
    parser = argparse.ArgumentParser('VQPy Demo!')
    parser.add_argument('--path', help='path to video')
    parser.add_argument(
        "--save_folder",
        default=None,
        help="the folder to save the final result",
    )
    parser.add_argument(
        "-d",
        "--pretrained_model_dir",
        help="Directory to pretrained models")
    return parser


class Person(vqpy.VObjBase):
    pass


class CountPersonOnCrosswalk(vqpy.QueryBase):

    @staticmethod
    def set_output_configs() -> vqpy.OutputConfig:
        return vqpy.OutputConfig(
            # output_frame_vobj_num=True,
            output_total_vobj_num=True
            )

    @staticmethod
    def setting() -> vqpy.VObjConstraint:

        CROSSWALK_REGION_1 = [(731, 554), (963, 564), (436, 1076), (14, 1076)]
        CROSSWALK_REGION_2 = [(1250, 528), (1292, 473),
                              (1839, 492), (1893, 547)]
        CROSSWALK_REGIONS = [CROSSWALK_REGION_1, CROSSWALK_REGION_2]

        def get_bottom_central_point(tlbr):
            x = (tlbr[0] + tlbr[2]) / 2
            y = tlbr[3]
            return (x, y)

        def on_crosswalk(tlbr):
            from shapely.geometry import Point, Polygon
            bottom_central_point = get_bottom_central_point(tlbr)
            point = Point(bottom_central_point)
            for region in CROSSWALK_REGIONS:
                poly = Polygon(region)
                if point.within(poly):
                    return True
            return False

        filter_cons = {"__class__": lambda x: x == Person,
                       "tlbr": on_crosswalk}
        select_cons = {"track_id": None,
                       }
        return vqpy.VObjConstraint(filter_cons=filter_cons,
                                   select_cons=select_cons,
                                   filename='on_crosswalk')


if __name__ == '__main__':
    args = make_parser().parse_args()
    register("yolox", YOLOXDetector, "yolox_x.pth")
    vqpy.launch(cls_name=vqpy.COCO_CLASSES,
                cls_type={"person": Person},
                tasks=[CountPersonOnCrosswalk()],
                video_path=args.path,
                save_folder=args.save_folder,
                detector_name="yolox",
                detector_model_dir=args.pretrained_model_dir)
