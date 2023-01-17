"""This is a demo VQPy implementation listing and storing all red moving
vehicles to a json file."""

import argparse
import numpy as np
import vqpy
from getcolor import get_image_color  # noqa: F401


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


class Vehicle(vqpy.VObjBase):
    """The class of vehicles"""
    required_fields = ['class_id', 'tlbr']

    @vqpy.property()
    @vqpy.postproc({'majority': 100})
    def license_plate(self):
        """The license plate of the vehicle"""
        return self.infer('license_plate', {'license_plate': 'openalpr'})


class ListMovingVehicle(vqpy.QueryBase):
    """The class obtaining all moving vehicles"""
    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        filter_cons = {'__class__': lambda x: x == Vehicle,
                       'bbox_velocity': lambda x: x >= 0.1}
        select_cons = {'track_id': None,
                       'license_plate': None}
        return vqpy.VObjConstraint(filter_cons=filter_cons,
                                   select_cons=select_cons,
                                   filename='moving')


class ListRedMovingVehicle(ListMovingVehicle):
    """The class obtaining all red moving vehicles"""
    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        import webcolors

        def rgb_is_red(color):
            color = np.asarray(color)
            return color[0] ** 2 > sum(color * color) * 0.7

        select_cons = {'track_id': None,
                       'license_plate': None,
                       'major_color_rgb': webcolors.rgb_to_name}
        filter_cons = {'major_color_rgb': rgb_is_red}
        return vqpy.VObjConstraint(select_cons=select_cons,
                                   filter_cons=filter_cons,
                                   filename="redmoving")


if __name__ == '__main__':
    args = make_parser().parse_args()
    vqpy.launch(cls_name=vqpy.COCO_CLASSES,
                cls_type={"car": Vehicle, "truck": Vehicle},
                tasks=[ListRedMovingVehicle()],
                video_path=args.path,
                save_folder=args.save_folder,
                detector_model_dir=args.pretrained_model_dir)
