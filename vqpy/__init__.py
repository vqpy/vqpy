import argparse
import os
from typing import List
from tqdm import tqdm
from loguru import logger

from vqpy.video_loader import FrameStream
from vqpy.predictor import YOLOXPredictor
from vqpy.tracker import ByteTracker
from vqpy.basics import MultiTracker
from vqpy.visualize import Visualizer
from vqpy.objects import property, stateful
from vqpy.functions import infer
from vqpy.objects import VObjBase
from vqpy.database import QueryBase, vobj_select, vobj_filter

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
    
    # YOLOX model setup
    yolox_predictor = YOLOXPredictor(device="gpu", fp16=True)
    # Bytetracker setup
    byte_tracker = MultiTracker(ByteTracker, stream, cls_name, cls_type)
    
    for frame_id in tqdm(range(1, stream.n_frames + 1)):
        frame = stream.next()
        outputs = yolox_predictor.inference(frame)
        tracked_tracks, _ = byte_tracker.update(outputs)
        for worker in workers:
            res = worker.apply(tracked_tracks)
            # TODO: add downstream functions for worker processing
            # if res != []: print(res)
        if args.save_result:
            visual.vis(frame, tracked_tracks, yolox_predictor.cls_names)
