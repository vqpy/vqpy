# precompute detection with YOLOX detection

import os
import argparse
from loguru import logger
from tqdm import tqdm
import pickle

import vqpy
from vqpy.operator.detector import setup_detector
from vqpy.operator.video_reader import FrameStream


def launch(
    cls_name,
    video_path: str,
    save_folder: str = None,
    detector_name: str = "yolox",
):
    logger.info(
        f"VQPy detection precompute I/O Setting: \
                  video_path={video_path}, save_folder={save_folder}"
    )
    video_name = os.path.basename(video_path).split(".")[0]
    stream = FrameStream(video_path)
    detector_name, detector = setup_detector(cls_name, detector_name=detector_name)

    precomputed = {}

    for frame_id in tqdm(range(1, stream.n_frames + 1)):
        frame_image = stream.next()
        outputs = detector.inference(frame_image)
        precomputed[frame_id] = outputs

    os.makedirs(save_folder, exist_ok=True)
    filename = f"{video_name}_{detector_name}.pkl"
    save_path = os.path.join(save_folder, filename)
    with open(save_path, "wb") as f:
        pickle.dump(precomputed, f)
    logger.info("Done!")


def make_parser():
    parser = argparse.ArgumentParser("VQPy detection precompute!")
    parser.add_argument("--path", help="path to video")
    parser.add_argument(
        "--save_folder",
        default=None,
        help="the folder to save precomputation result",
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    launch(
        cls_name=vqpy.COCO_CLASSES,
        video_path=args.path,
        save_folder=args.save_folder,
    )
