import cv2
from loguru import logger

# TODO: support different types of video streams


class FrameStream:
    output_fields = ['frame', 'frame_id', 'frame_width', 'frame_height', 'fps']

    def __init__(self, path):
        self._cap = cv2.VideoCapture(path)
        self.frame_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        self.frame_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_id = -1
        self.frame = None
        logger.info(f"Parameters of video is width={self.frame_width}, \
                      height={self.frame_height}, fps={self.fps}")
        self._objdatas = None

    def next(self):
        self.frame_id += 1
        ret_val, self.frame = self._cap.read()
        if not ret_val:
            logger.info(f"Failed to load frame stream with id of "
                        f"{self.frame_id}")
            raise IOError
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord('Q'):
            raise KeyboardInterrupt
        return self.frame
