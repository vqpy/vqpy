import os

from loguru import logger
from tqdm import tqdm
import pickle

from vqpy.operator.detector import setup_detector
from vqpy.operator.video_reader import FrameStream


def launch(
    cls_name,
    video_path: str,
    save_folder: str = None,
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
    logger.info(
        f"VQPy Launch I/O Setting: \
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
