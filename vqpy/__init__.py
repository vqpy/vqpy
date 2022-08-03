import argparse
import os
from typing import List

from loguru import logger
from tqdm import tqdm

from vqpy.base.query import QueryBase
from vqpy.detector import setup_detector
from vqpy.feat.database import vobj_argmin, vobj_filter, vobj_select
from vqpy.feat.feat import access_data, postproc, property, stateful
from vqpy.function import infer
from vqpy.impl.multiclass_tracker import \
    MultiTracker as default_surface_tracker
from vqpy.impl.vobj_base import VObjBase
from vqpy.tracker import setup_ground_tracker
from vqpy.utils.classes import COCO_CLASSES
from vqpy.utils.video import FrameStream
from vqpy.visualize import Visualizer


def make_parser():
    parser = argparse.ArgumentParser("VQPy Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--path", default="./video_dataset/southampton/raw_000.mp4", help="path to video")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    return parser

def prepare_info(img, id, fps, padding_info={}):
    img_info = padding_info.copy()
    img_info["id"] = id
    img_info["fps"] = fps

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img
    
    return img_info

def launch(cls_name, cls_type, workers: List[QueryBase]):
    args = make_parser().parse_args()
    logger.info("Args: {}".format(args))
    
    stream = FrameStream(args.path)
    if args.save_result:
        visual = Visualizer(args, stream)
    
    detector = setup_detector(device="gpu", fp16=True)
    tracker = default_surface_tracker(setup_ground_tracker, stream, cls_name, cls_type)
    # worker setup
    for worker in workers: worker.attach(stream)
    
    for frame_id in tqdm(range(1, stream.n_frames + 1)):
        frame = stream.next()
        outputs = detector.inference(frame)
        tracked_tracks, _ = tracker.update(outputs)
        for worker in workers:
            worker.apply(tracked_tracks)
        if args.save_result:
            visual.vis(frame, tracked_tracks, detector.cls_names)
