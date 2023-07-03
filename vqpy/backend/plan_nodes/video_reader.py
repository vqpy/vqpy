from vqpy.backend.operator.video_reader import VideoReader
from vqpy.backend.plan_nodes.base import AbstractPlanNode


class VideoReaderNode(AbstractPlanNode):

    def __init__(self):
        super().__init__()

    def to_operator(self, lauch_args: dict):
        return VideoReader(lauch_args["video_path"])
