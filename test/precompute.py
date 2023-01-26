# loitering, yolox

import argparse
import vqpy
from fake_launch import launch


def make_parser():
    parser = argparse.ArgumentParser("VQPy Demo!")
    parser.add_argument("--path", help="path to video")
    parser.add_argument(
        "--save_folder",
        default=None,
        help="the folder to save the final result",
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    launch(
        cls_name=vqpy.COCO_CLASSES,
        video_path=args.path,
        save_folder=args.save_folder,
    )
