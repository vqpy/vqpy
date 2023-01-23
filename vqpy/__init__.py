import json
import os
from typing import List, Dict

from loguru import logger
from tqdm import tqdm

from vqpy.operator.detector.base import DetectorBase  # noqa: F401
from vqpy.query.base import QueryBase
from vqpy.operator.tracker.base import GroundTrackerBase  # noqa: F401
from vqpy.operator.detector import setup_detector
from vqpy.obj.vobj.wrappers import property, stateful, postproc, cross_vobj_property  # noqa: F401,E501
from vqpy.property_lib.wrappers import vqpy_func_logger  # noqa: F401
from vqpy.operator.tracker.multiclass_tracker import MultiTracker
from vqpy.obj.vobj.base import VObjBase
from vqpy.query.vobj_constraint import VObjConstraint  # noqa: F401
from vqpy.obj.frame import Frame
from vqpy.operator.tracker import setup_ground_tracker
from vqpy.class_names.coco import COCO_CLASSES  # noqa: F401
from vqpy.operator.video_reader import FrameStream
from . import utils  # noqa: F401


def launch(cls_name,
           cls_type: Dict[str, VObjBase],
           tasks: List[QueryBase],
           video_path: str,
           save_folder: str = None,
           save_freq: int = 10,
           detector_name: str = "yolox",
           ):
    """Launch the VQPy tasks with specific setting.
    Args:
        cls_name: the detector classification result classes.
        cls_type: the mapping from each class to corresponding VObj.
        tasks (List[QueryBase]): the list of queries to apply.
        video_path (str): the path of the queried video.
        save_folder: the folder to save final result.
        save_freq: the frequency of save when processing.
        detector_model_dir: the directory for all pretrained detectors.
        detector_name: the specific detector name you desire to use.
    """
    logger.info(f"VQPy Launch I/O Setting: \
                  video_path={video_path}, save_folder={save_folder}")
    video_name = os.path.basename(video_path).split(".")[0]
    stream = FrameStream(video_path)
    detector_name, detector = setup_detector(cls_name,
                                             detector_name=detector_name)
    # Now tracking is always performed by track each class separately
    frame = Frame(stream)
    tracker = MultiTracker(setup_ground_tracker, cls_name, cls_type)
    for task in tasks:
        task.vqpy_init()

    tag = stream.n_frames
    for frame_id in tqdm(range(1, stream.n_frames + 1)):
        frame_image = stream.next()
        outputs = detector.inference(frame_image)
        frame = tracker.update(outputs, frame)
        for task in tasks:
            task.vqpy_update(frame)

        if frame_id * save_freq >= tag and save_folder:
            if tag == stream.n_frames:
                os.makedirs(save_folder, exist_ok=True)
            for task in tasks:
                task_name = task.get_setting().filename
                filename = f"{video_name}_{task_name}_{detector_name}.json"
                save_path = os.path.join(save_folder, filename)
                with open(save_path, 'w') as f:
                    json.dump(task.vqpy_getdata(), f)
            tag += stream.n_frames
    logger.info("Done!")
