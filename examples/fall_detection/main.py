import sys
import os
import vqpy

import torch
import numpy as np
import argparse

# importing pose detection models
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detect')
)
from PoseEstimateLoader import SPPE_FastPose  # noqa: E402
from ActionsEstLoader import TSSTG  # noqa: E402


def make_parser():
    parser = argparse.ArgumentParser('VQPy Demo!')
    parser.add_argument('--path', help='path to video')
    parser.add_argument(
        "--save_folder",
        default=None,
        help="the folder to save the final result",
    )
    return parser


class Person(vqpy.VObjBase):
    required_fields = ['class_id', 'tlbr']

    # default values, to be assigned in main()
    pose_model = None
    action_model = None

    @vqpy.property()
    @vqpy.stateful(30)
    def keypoints(self):
        # per-frame property, but tracker can return objects
        # not in the current frame
        image = self._ctx.frame
        tlbr = self.getv('tlbr')
        if tlbr is None:
            return None
        return Person.pose_model.predict(image, torch.tensor([tlbr]))

    @vqpy.property()
    def pose(self) -> str:
        keypoints_list = []
        for i in range(-self._track_length, 0):
            keypoint = self.getv('keypoints', i)
            if keypoint is not None:
                keypoints_list.append(keypoint)
            if len(keypoints_list) >= 30:
                break
        if len(keypoints_list) < 30:
            return 'unknown'
        pts = np.array(keypoints_list, dtype=np.float32)
        out = Person.action_model.predict(pts, self._ctx.frame.shape[:2])
        action_name = Person.action_model.class_names[out[0].argmax()]
        return action_name


class FallDetection(vqpy.QueryBase):
    """The class obtaining all fallen person"""
    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        filter_cons = {'__class__': lambda x: x == Person,
                       'pose': lambda x: x == "Fall Down"}
        select_cons = {'track_id': None,
                       'tlbr': lambda x: str(x)}
        return vqpy.VObjConstraint(filter_cons=filter_cons,
                                   select_cons=select_cons,
                                   filename='fall')


if __name__ == '__main__':
    args = make_parser().parse_args()
    model_dir = args.pretrained_model_dir
    pose_model = SPPE_FastPose(
        'resnet50', 224, 160, device='cuda',
        weights_file=os.path.join(
            os.path.abspath(model_dir), "fast_res50_256x192.pth"
        )
    )
    action_model = TSSTG(
        weight_file=os.path.join(os.path.abspath(model_dir), "tsstg-model.pth")
    )
    Person.pose_model = pose_model
    Person.action_model = action_model
    vqpy.launch(
        cls_name=vqpy.COCO_CLASSES,
        cls_type={"person": Person},
        tasks=[FallDetection()],
        video_path=args.path,
        save_folder=args.save_folder,
        detector_model_dir=args.pretrained_model_dir
    )
