from vqpy.backend.operator.video_reader import VideoReader
from vqpy.backend.plan_nodes.frame_filter import VObjFrameFilterNode
from vqpy.backend.plan_nodes.output_formatter import FrameOutputNode
from vqpy.backend.plan_nodes.tracker import TrackerNode
from vqpy.backend.plan_nodes.vobj_filter import VObjFilterNode
from vqpy.backend.plan_nodes.vobj_projector import ProjectorNode
from vqpy.backend.plan_nodes.base import AbstractPlanNode
from vqpy.backend.plan_nodes.object_detector import ObjectDetectorNode
from vqpy.backend.plan_nodes.video_reader import VideoReaderNode
from vqpy.backend.plan_nodes.vobj_projector import ProjectionField
from vqpy.frontend.query import QueryBase
from vqpy.frontend.vobj.predicates import Predicate, IsInstance
from vqpy.frontend.vobj.property import Property, BuiltInProperty


class Planner:

    def print_plan(self, node: AbstractPlanNode = None):
        print(node)
        if node.get_prev() is not None:
            self.print_plan(node.get_prev())

    def parse(self, query_obj: QueryBase):
        input_node = VideoReaderNode()
        output_node = self._create_object_detector_node(query_obj, input_node)
        output_node = self._create_tracker_node(query_obj, output_node)
        output_node = self._create_vobj_class_filter_node(query_obj,
                                                          output_node)
        output_node, map = self._create_pre_filter_projector(query_obj,
                                                             output_node)
        output_node = self._create_frame_filter_node(query_obj, output_node)
        output_node = self._create_frame_output_projector(query_obj,
                                                          output_node, map)
        output_node = self._create_frame_output_formatter(query_obj,
                                                          output_node)
        return output_node

    def _create_object_detector_node(self, query_obj: QueryBase, input_node):
        frame_constraints = query_obj.frame_constraint()
        assert isinstance(frame_constraints, Predicate)
        vobjs = frame_constraints.get_vobjs()
        assert len(vobjs) == 1, "Only support one vobj in the predicate"
        vobj = list(vobjs)[0]
        class_names = vobj.class_name
        detector_name = vobj.object_detector
        detector_kwargs = vobj.detector_kwargs
        return input_node.set_next(
            ObjectDetectorNode(class_names=class_names,
                               detector_name=detector_name,
                               detector_kwargs=detector_kwargs)
        )

    def _create_tracker_node(self, query_obj: QueryBase, input_node):
        frame_constraints = query_obj.frame_constraint()
        assert isinstance(frame_constraints, Predicate)
        vobjs = frame_constraints.get_vobjs()
        assert len(vobjs) == 1, "Only support one vobj in the predicate"
        vobj = list(vobjs)[0]
        class_name = vobj.class_name
        return input_node.set_next(
            TrackerNode(class_name=class_name)
        )

    def _create_pre_filter_projector(self, query_obj: QueryBase, input_node):
        frame_constraints = query_obj.frame_constraint()

        node = input_node

        vobj_properties_map = dict()

        if isinstance(frame_constraints, Predicate):
            vobjs = frame_constraints.get_vobjs()
            assert len(vobjs) == 1, "Only support one vobj in the predicate"
            vobj = list(vobjs)[0]
            vobj_properties = frame_constraints.get_vobj_properties()
            for p in vobj_properties:
                projector_node = ProjectorNode(
                    class_name=vobj.class_name,
                    projection_field=ProjectionField(
                        field_name=p.name,
                        field_func=p,
                        dependent_fields=p.inputs,
                    ),
                    filter_index=0
                )
                node = node.set_next(projector_node)
            vobj_properties_map[vobj] = vobj_properties

        return node, vobj_properties_map

    def _create_vobj_class_filter_node(self, query_obj: QueryBase, input_node):
        frame_constraints = query_obj.frame_constraint()
        node = input_node

        vobjs = frame_constraints.get_vobjs()
        assert len(vobjs) == 1, "Only support one vobj in the predicate"
        vobj = list(vobjs)[0]
        node = node.set_next(
            VObjFilterNode(predicate=IsInstance(vobj), filter_index=0))
        return node

    def _create_frame_filter_node(self, query_obj: QueryBase, input_node):
        predicate = query_obj.frame_constraint()

        output_node = input_node.set_next(
            VObjFilterNode(predicate=predicate, filter_index=0))
        output_node = output_node.set_next(VObjFrameFilterNode(filter_index=0))
        return output_node

    def _create_frame_output_projector(self, query_vobj: QueryBase,
                                       input_node,
                                       vobj_properties_map: dict):
        existing_vobj_properties = vobj_properties_map.copy()
        frame_output = query_vobj.frame_output()
        if isinstance(frame_output, Property):
            frame_output = [frame_output]
        for prop in frame_output:
            vobj = prop.get_vobjs()
            assert len(vobj) == 1, "Only support one vobj for vobj_property."
            vobj = list(vobj)[0]
            existing_properties = existing_vobj_properties[vobj]
            if not isinstance(prop, BuiltInProperty):
                if all([prop.func != ep.func for ep in existing_properties]):
                    projector_node = ProjectorNode(
                        class_name=vobj.class_name,
                        projection_field=ProjectionField(
                            field_name=prop.name,
                            field_func=prop,
                            dependent_fields=prop.inputs,
                        ),
                        filter_index=0
                    )
                    input_node = input_node.set_next(projector_node)
                    existing_properties.append(prop)
        return input_node

    def _create_frame_output_formatter(self, query_vobj: QueryBase,
                                       input_node):
        frame_output = query_vobj.frame_output()
        if isinstance(frame_output, Property):
            frame_output = [frame_output]
        property_mappings = dict()
        vobj_name_mappings = dict()

        # todo: support multiple vobjs
        filter_index = 0
        vobj = frame_output[0].get_vobjs()
        assert len(vobj) == 1, "Only support one vobj for vobj_property."
        vobj = list(vobj)[0]
        class_name = vobj.class_name
        name = vobj.name
        vobj_name_mappings[filter_index] = name
        property_mappings[filter_index] = {
            class_name: [prop.name for prop in frame_output]
        }
        frame_output_formatter = FrameOutputNode(
            filter_index_to_class_name_to_property_names=property_mappings,
            filter_index_to_vobj_name=vobj_name_mappings
        )
        return input_node.set_next(frame_output_formatter)


def add_video_metadata(launch_args: dict):
    assert "video_path" in launch_args
    video_path = launch_args["video_path"]
    video_reader = VideoReader(video_path=video_path)
    video_metadata = video_reader.get_metadata()
    launch_args.update(video_metadata)
    video_reader.close()
    return launch_args