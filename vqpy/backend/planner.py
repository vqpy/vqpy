from vqpy.backend.operator.video_reader import VideoReader
from vqpy.backend.plan_nodes.frame_filter import create_frame_filter_node
from vqpy.backend.plan_nodes.output_formatter import create_frame_output_formatter
from vqpy.backend.plan_nodes.tracker import create_tracker_node
from vqpy.backend.plan_nodes.vobj_filter import (
    create_vobj_class_filter_node,
    create_vobj_filter_node
)
from vqpy.backend.plan_nodes.vobj_projector import (
    create_frame_output_projector,
    create_pre_filter_projector
)
from vqpy.backend.plan_nodes.base import AbstractPlanNode
from vqpy.backend.plan_nodes.object_detector import create_object_detector_node
from vqpy.backend.plan_nodes.video_reader import VideoReaderNode
from vqpy.frontend.query import QueryBase


class Planner:

    def print_plan(self, node: AbstractPlanNode = None):
        print(node)
        if node.get_prev() is not None:
            self.print_plan(node.get_prev())

    def parse(self, query_obj: QueryBase):
        input_node = VideoReaderNode()
        output_node = create_object_detector_node(query_obj, input_node)
        output_node = create_tracker_node(query_obj, output_node)
        output_node = create_vobj_class_filter_node(query_obj,
                                                    output_node)
        output_node, map = create_pre_filter_projector(query_obj,
                                                       output_node)
        output_node = create_vobj_filter_node(query_obj, output_node)
        output_node = create_frame_filter_node(query_obj, output_node)
        output_node = create_frame_output_projector(query_obj,
                                                    output_node, map)
        output_node = create_frame_output_formatter(query_obj,
                                                    output_node)
        return output_node


def add_video_metadata(launch_args: dict):
    assert "video_path" in launch_args
    video_path = launch_args["video_path"]
    video_reader = VideoReader(video_path=video_path)
    video_metadata = video_reader.get_metadata()
    launch_args.update(video_metadata)
    video_reader.close()
    return launch_args
