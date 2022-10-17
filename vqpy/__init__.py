import json
import os
from typing import List, Dict

from loguru import logger
from tqdm import tqdm

from .base.detector import DetectorBase  # noqa: F401
from .base.query import QueryBase
from .base.ground_tracker import GroundTrackerBase  # noqa: F401
from .detector import setup_detector
from .feat.feat import property, stateful, postproc  # noqa: F401
from .function import infer  # noqa: F401
from .function.logger import vqpy_func_logger  # noqa: F401
from .impl.multiclass_tracker import MultiTracker
from .impl.vobj_base import VObjBase
from .impl.vobj_constraint import VObjConstraint  # noqa: F401
from .tracker import setup_ground_tracker
from .utils.classes import COCO_CLASSES  # noqa: F401
from .utils.video import FrameStream


def launch(cls_name,
           cls_type: Dict[str, VObjBase],
           tasks: List[QueryBase],
           video_path: str,
           save_folder: str = None,
           save_freq: int = 10,
           detector_model_path: str = None):
    """Launch the VQPy tasks with specific setting.
    Args:
        cls_name: the detector classification result classes.
        cls_type: the mapping from each class to corresponding VObj.
        tasks (List[QueryBase]): the list of queries to apply.
        video_path (str): the path of the queried video.
        save_folder: the folder to save final result.
        save_freq: the frequency of save when processing.
        detector_model_path: the pretrained detector path.
    """
    logger.info(f"VQPy Launch I/O Setting: \
                  video_path={video_path}, save_folder={save_folder}")
    stream = FrameStream(video_path)
    detector = setup_detector(cls_name, detector_model_path)
    # Now tracking is always performed by track each class separately
    tracker = MultiTracker(setup_ground_tracker, stream, cls_name, cls_type)
    for task in tasks:
        task.vqpy_init()

    tag = stream.n_frames
    for frame_id in tqdm(range(1, stream.n_frames + 1)):
        frame = stream.next()
        outputs = detector.inference(frame)
        tracked_tracks, _ = tracker.update(outputs)
        for task in tasks:
            task.vqpy_update(frame_id, tracked_tracks)

        if frame_id * save_freq >= tag and save_folder:
            if tag == stream.n_frames:
                os.makedirs(save_folder, exist_ok=True)
            for task in tasks:
                filename = task.get_setting().filename + '.json'
                save_path = os.path.join(save_folder, filename)
                with open(save_path, 'w') as f:
                    json.dump(task.vqpy_getdata(), f)
            tag += stream.n_frames
