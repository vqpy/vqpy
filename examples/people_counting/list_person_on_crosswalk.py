import argparse
import numpy as np
import vqpy
from vqpy.function.logger import vqpy_func_logger
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


@vqpy_func_logger(["frame"], ["crosswalk_frame"], [], required_length=1)
def filter_crosswalk_color(obj, frame):
    rgb_L = np.array([160, 160, 160])
    rgb_R = np.array([200, 200, 200])
    frame = np.logical_and(frame >= rgb_L, frame <= rgb_R)
    frame = np.logical_and(np.logical_and(frame[:, :, 0], frame[:, :, 1]),
                           frame[:, :, 2]).astype(int)
    return [frame]


class Pedestrian(vqpy.VObjBase):
    """The class of pedestrians"""
    required_fields = ['class_id', 'tlbr']

    @vqpy.property()
    @vqpy.stateful(2)
    def first_on_crosswalk(self):
        # top left, bottom right
        if self.getv('once_on_crosswalk', -2):
            return False
        tlbr = self.getv('tlbr')
        if tlbr is None:
            return False
        crosswalk = self.getv('crosswalk_frame')
        left, right = int(tlbr[0]), int(tlbr[2])
        row = min(int(tlbr[3] + 1), int(self.getv('frame_height')) - 1)
        return np.mean(crosswalk[row, left:right+1]) >= .5

    @vqpy.property()
    @vqpy.stateful(2)
    def once_on_crosswalk(self):
        return (self.getv('once_on_crosswalk', -2)
                or self.getv('first_on_crosswalk'))


class ListPersonOnCrosswalk(vqpy.QueryBase):
    """The class obtaining all moving vehicles"""
    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        filter_cons = {'__class__': lambda x: x == Pedestrian,
                       'first_on_crosswalk': lambda x: x}
        select_cons = {'track_id': None}
        return vqpy.VObjConstraint(filter_cons=filter_cons,
                                   select_cons=select_cons,
                                   filename='count')


if __name__ == '__main__':
    args = make_parser().parse_args()
    register("yolox", YOLOXDetector, "yolox_x.pth")
    vqpy.launch(cls_name=vqpy.COCO_CLASSES,
                cls_type={"person": Pedestrian},
                tasks=[ListPersonOnCrosswalk()],
                video_path=args.path,
                save_folder=args.save_folder,
                detector_name="yolox",
                detector_model_dir=args.pretrained_model_dir)
