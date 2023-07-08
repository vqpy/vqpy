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
from . import query  # noqa: F401
from vqpy.backend.operator import CustomizedVideoReader


def launch(
    cls_name,
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
    detector_name, detector = setup_detector(
        cls_name, detector_name=detector_name
    )
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
                with open(save_path, "w") as f:
                    json.dump(task.vqpy_getdata(), f)
            tag += stream.n_frames

    # reset Tracker after we finish
    tracker.reset()

    logger.info("Done!")


def init(
    query_obj,
    video_path: str = None,
    custom_video_reader: CustomizedVideoReader = None,
    additional_frame_fields: List[str] = None,
    output_per_frame_results: bool = False,
    verbose: bool = True,
):
    """
    Args:
        query_obj: the query object to apply.
        video_path: the path of the video to query on.
        custom_video_reader: the custom video reader to use. If not None, will
            ignore video_path. Default: None. Note that fps must be provided
            if custom_video_reader is not None.
        additional_frame_fields: the additional frame fields to output.
        output_per_frame_results: whether to output per frame results. Default:
            False, which only output the frames containing the objects that
            meet the query constraints. If True, will output all frames, where
            frames without objects that meet the query constraints will have
            results as an empty list.
        verbose: whether to print the progress. Default: True.
    """
    from vqpy.backend import Planner, Executor

    # input check
    if custom_video_reader is None:
        if video_path is None:
            raise ValueError(
                "video_path must be provided if custom_video_reader is"
                "None"
            )
        if not os.path.exists(video_path):
            raise ValueError(f"video_path {video_path} does not exist")
    else:
        if not isinstance(custom_video_reader, CustomizedVideoReader):
            raise ValueError(
                "custom_video_reader must be an instance of"
                "CustomizedVideoReader, got"
                f" {type(custom_video_reader)}"
            )

    planner = Planner()
    launch_args = {
        "video_path": video_path,
        "query_name": query_obj.__class__.__name__,
    }
    root_plan_node = planner.parse(
        query_obj, custom_video_reader=custom_video_reader,
        additional_frame_fields=additional_frame_fields,
        output_per_frame_results=output_per_frame_results,
    )
    if verbose:
        planner.print_plan(root_plan_node)
    executor = Executor(
        root_plan_node, launch_args, custom_video_reader=custom_video_reader
    )
    return executor


def run(
    executor,
    save_folder: str = None,
    print_results: bool = True,
):
    """
    Args:
        executor: the executor to run the query.
        save_folder: the folder to save query result.
            If None, will print to stdout. Default: None.
            If not None, will save to json file with the name of
            {query_name}.json in the save_folder.
        print_result: whether to print the result. Default: True.
    """

    result = executor.execute()
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        filename = executor.launch_args["query_name"] + ".json"
        save_path = os.path.join(save_folder, filename)
        print(f"Saving result to {save_path}")
        with open(save_path, "w") as f:
            for res in result:
                json.dump(res, f, cls=utils.NumpyEncoder)
                if print_results:
                    print(res)
        print(f"Done! Result saved to {save_path}")
    elif print_results:
        for res in result:
            print(res)
    return result
