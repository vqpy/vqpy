import argparse
import json
import os
import time
from typing import List

from loguru import logger
from tqdm import tqdm

from .base.query import QueryBase
from .database import VObjConstraint  # noqa: F401
from .detector import setup_detector
from .feat.feat import property, stateful, postproc  # noqa: F401
from .function import infer  # noqa: F401
from .function.logger import vqpy_func_logger  # noqa: F401
from .impl.multiclass_tracker import MultiTracker as _dtracker
from .impl.vobj_base import VObjBase  # noqa: F401
from .tracker import setup_ground_tracker
from .utils.classes import COCO_CLASSES  # noqa: F401
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
    tracker = _dtracker(setup_ground_tracker, stream, cls_name, cls_type)
    for worker in workers:
        worker._begin_query()
    for frame_id in tqdm(range(1, stream.n_frames + 1)):
        frame = stream.next()
        outputs = detector.inference(frame)
        tracked_tracks, _ = tracker.update(outputs)
        for worker in workers:
            worker._update_query(frame_id, tracked_tracks)

    if args.save_result:
        folder_name = os.path.join("./vqpy_outputs", args.experiment_name)
        os.makedirs(folder_name, exist_ok=True)
        current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        save_folder = os.path.join(folder_name, current_time)
        os.makedirs(save_folder, exist_ok=True)
        for worker in workers:
            filename = worker._get_setting().filename + '.json'
            save_path = os.path.join(save_folder, filename)
            with open(save_path, 'w') as f:
                json.dump(worker._end_query(), f)
