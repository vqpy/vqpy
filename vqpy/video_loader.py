from loguru import logger
import cv2
import os

from vqpy.funcutils import vqpy_logger

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
        logger.info("Parameters of video is width={}, height={}, fps={}".format(self.frame_width, self.frame_height, self.fps))
    
    @vqpy_logger
    def next(self):
        self.frame_id += 1
        ret_val, self.frame = self._cap.read()
        if not ret_val:
            logger.info("Failed to load frame stream")
            raise IOError
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord('Q'):
            raise KeyboardInterrupt
        return self.frame