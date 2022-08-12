import argparse
import os, time, json
from typing import List

from loguru import logger
from tqdm import tqdm

from .base.query import QueryBase
from .detector import setup_detector
from .database import VObjConstraint
from .feat.feat import postproc, property, stateful
from .function import infer
from .function.logger import vqpy_func_logger
from .impl.multiclass_tracker import \
    MultiTracker as default_surface_tracker
from .impl.vobj_base import VObjBase
from .tracker import setup_ground_tracker
from .utils.classes import COCO_CLASSES
from .utils.video import FrameStream

def make_parser():
    parser = argparse.ArgumentParser("VQPy Demo!")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--path", help="path to video")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    return parser

def launch(cls_name, cls_type, workers: List[QueryBase]):
    args = make_parser().parse_args()
    logger.info("Args: {}".format(args))
    
    stream = FrameStream(args.path)
    
    detector = setup_detector(device="gpu", fp16=True)
    tracker = default_surface_tracker(setup_ground_tracker, stream, cls_name, cls_type)
    
    for worker in workers: worker._begin_query()
    
    for frame_id in tqdm(range(1, stream.n_frames + 1)):
        frame = stream.next()
        outputs = detector.inference(frame)
        tracked_tracks, _ = tracker.update(outputs)
        for worker in workers:
            worker._update_query(frame_id, tracked_tracks)
    
    if args.save_result:
        for worker in workers:
            folder_name = os.path.join("./vqpy_outputs", args.experiment_name)
            os.makedirs(folder_name, exist_ok=True)
            save_folder = os.path.join(folder_name, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, worker._get_setting().filename + '.json')
            with open(save_path, 'w') as f:
                json.dump(worker._end_query(), f)
