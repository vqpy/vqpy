import cv2
from loguru import logger
from vqpy.backend.operator.base import Operator
from vqpy.backend.frame import Frame


class VideoReader(Operator):
    def __init__(self, video_path):
        self._cap = cv2.VideoCapture(video_path)
        self.frame_id = -1
        self.metadata = self.get_metadata()

    def get_metadata(self):
        frame_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        frame_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Metadata of video is width={frame_width}, \
                      height={frame_height}, fps={fps}, n_frames={n_frames}")
        metadata = dict(
            frame_width=frame_width,
            frame_height=frame_height,
            fps=fps,
            n_frames=n_frames,
        )
        return metadata

    def has_next(self) -> bool:
        if self.frame_id + 1 < self.metadata["n_frames"]:
            return True
        else:
            self.close()
            return False

    def next(self) -> Frame:
        self.frame_id += 1
        ret_val, frame_image = self._cap.read()
        if not ret_val:
            logger.info(f"Failed to load frame stream with id of "
                        f"{self.frame_id}")
            raise IOError
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord('Q'):
            raise KeyboardInterrupt

        frame = Frame(video_metadata=self.metadata,
                      id=self.frame_id,
                      image=frame_image)
        return frame

    def close(self):
        self._cap.release()
