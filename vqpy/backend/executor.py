from vqpy.backend.operator import CustomizedVideoReader
from vqpy.backend.operator.video_reader import VideoReader


def add_video_metadata(
    launch_args: dict, custom_video_reader: CustomizedVideoReader = None
):
    if custom_video_reader is not None:
        video_metadata = custom_video_reader.get_metadata()
    else:
        video_path = launch_args["video_path"]
        assert video_path is not None
        video_reader = VideoReader(video_path=video_path)
        video_metadata = video_reader.get_metadata()
        video_reader.close()
    launch_args.update(video_metadata)
    return launch_args


class Executor:
    def __init__(
        self,
        root_plan_node,
        launch_args: dict,
        custom_video_reader: CustomizedVideoReader = None,
    ):
        self.root_plan_node = root_plan_node
        self.launch_args = add_video_metadata(
            launch_args, custom_video_reader=custom_video_reader
        )
        self.root_operator = root_plan_node.to_operator(self.launch_args)

    def execute(self):
        while self.root_operator.has_next():
            result = self.root_operator.next()
            yield result
