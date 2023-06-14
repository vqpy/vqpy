from vqpy.backend.operator.video_reader import VideoReader
from vqpy.backend.operator.object_detector import ObjectDetector
from vqpy.backend.operator.vobj_filter import VObjFilter
from vqpy.backend.operator.frame_filter import VObjFrameFilter
from vqpy.backend.operator.tracker import Tracker
from vqpy.backend.operator.vobj_projector import VObjProjector
from vqpy.backend.operator.output_formatter import FrameOutputFormatter
from vqpy.frontend.query import QueryBase
from vqpy.frontend.vobj.predicates import Predicate, IsInstance
from vqpy.frontend.vobj.property import Property, BuiltInProperty
from abc import abstractmethod
from typing import Set, Union, Optional, Dict, Callable, Any, List


class AbstractPlanNode:

    def __init__(self):
        self.next = None
        self.prev = None

    def set_prev(self, plan_node):
        '''
        Set the plan_node as the current node's child node. The current node
        will consume or depends on the child node.

        return the child node
        '''
        self.prev = plan_node
        plan_node.next = self
        return plan_node

    def get_prev(self):
        return self.prev

    def set_next(self, plan_node):
        self.next = plan_node
        plan_node.prev = self
        return self.next

    def get_next(self):
        return self.next

    @abstractmethod
    def to_operator(self, lauch_args: dict):
        pass

    def __str__(self):
        return f"PlanNode({self.__class__.__name__},\n" \
                f"\tprev={self.prev.__class__.__name__},\n" \
                f"\tnext={self.next.__class__.__name__})"


class VideoReaderNode(AbstractPlanNode):

    def __init__(self):
        super().__init__()

    def to_operator(self, lauch_args: dict):
        return VideoReader(lauch_args["video_path"])


class ObjectDetectorNode(AbstractPlanNode):

    def __init__(self,
                 class_names: Union[str, Set[str]],
                 detector_name: Optional[str] = None,
                 detector_kwargs: dict = None):
        self.class_names = class_names
        self.detector_name = detector_name
        self.detector_kwargs = detector_kwargs \
            if detector_kwargs is not None else dict()
        super().__init__()

    def to_operator(self, launch_args: dict):
        return ObjectDetector(
            prev=self.prev.to_operator(launch_args),
            class_names=self.class_names,
            detector_name=self.detector_name,
            **self.detector_kwargs
        )


class ProjectionField:

    def __init__(self,
                 field_name: str,
                 field_func: Callable[[Dict], Any],
                 dependent_fields: Dict[str, int]):
        self.field_name = field_name
        self.field_func = field_func
        self.dependent_fields = dependent_fields


class ProjectorNode(AbstractPlanNode):

    def __init__(self,
                 class_name: str,
                 projection_field: ProjectionField,
                 filter_index: int):
        self.class_name = class_name
        self.projection_field = projection_field
        self.filter_index = filter_index
        super().__init__()

    def to_operator(self, launch_args: dict):
        return VObjProjector(
            prev=self.prev.to_operator(launch_args),
            property_name=self.projection_field.field_name,
            property_func=self.projection_field.field_func,
            dependencies=self.projection_field.dependent_fields,
            class_name=self.class_name,
            filter_index=self.filter_index
        )

    def __str__(self):
        return f"ProjectorNode(class_name={self.class_name}, \n" \
                f"\tproperty_name={self.projection_field.field_name}, \n" \
                f"\tfilter_index={self.filter_index}), \n" \
                f"\tdependencies={self.projection_field.dependent_fields}),\n"\
                f"\tprev={self.prev.__class__.__name__}), \n"\
                f"\text={self.next.__class__.__name__})"


class TrackerNode(AbstractPlanNode):

    def __init__(self,
                 class_name: str,
                 filter_index: Optional[int] = None,
                 tracker_name: str = "byte",
                 ):
        self.class_name = class_name
        self.filter_index = filter_index
        self.tracker_name = tracker_name
        super().__init__()

    def to_operator(self, launch_args: dict):
        # fps is for byte tracker
        return Tracker(
            prev=self.prev.to_operator(launch_args),
            class_name=self.class_name,
            filter_index=self.filter_index,
            tracker_name=self.tracker_name,
            fps=launch_args["fps"],
        )

    def __str__(self):
        return f"TrackerNode(class_name={self.class_name}, \n" \
            f"\tfilter_index={self.filter_index}, \n" \
            f"\ttracker_name={self.tracker_name}), \n" \
            f"\tprev={self.prev.__class__.__name__}), \n"\
            f"\tnext={self.next.__class__.__name__})"


class VObjFilterNode(AbstractPlanNode):

    def __init__(self,
                 predicate: Predicate,
                 filter_index):
        self.predicate = predicate
        self.filter_index = filter_index
        super().__init__()

    def to_operator(self, lauch_args: dict):
        return VObjFilter(
            prev=self.prev.to_operator(lauch_args),
            condition_func=self.predicate.generate_condition_function(),
            filter_index=self.filter_index

        )

    def __str__(self):
        return f"VObjFilterNode(predicate={self.predicate}, \n" \
               f"\tfilter_index={self.filter_index}), \n" \
               f"\tprev={self.prev.__class__.__name__}), \n"\
               f"\tnext={self.next.__class__.__name__})"


class VObjFrameFilterNode(AbstractPlanNode):

    def __init__(self, filter_index: int = 0):
        self.filter_index = filter_index
        super().__init__()

    def to_operator(self, lauch_args: dict):
        return VObjFrameFilter(prev=self.prev.to_operator(lauch_args),
                               vobj_filter_index=self.filter_index)

    def __str__(self):
        return f"VObjFrameFilterNode(filter_index={self.filter_index}), \n"\
            f"\tprev={self.prev.__class__.__name__}), \n" \
            f"\tnext={self.next.__class__.__name__})"


class FrameOutputNode(AbstractPlanNode):

    def __init__(self,
                 filter_index_to_class_name_to_property_names:
                 Dict[int, Dict[str, List[str]]],
                 filter_index_to_vobj_name: Dict[int, str]
                 ):
        self.props_mapping = filter_index_to_class_name_to_property_names
        self.vobj_names_mapping = filter_index_to_vobj_name
        super().__init__()

    def to_operator(self, lauch_args: dict):
        return FrameOutputFormatter(
            prev=self.prev.to_operator(lauch_args),
            filter_index_to_class_name_to_property_names=self.props_mapping,
            filter_index_to_vobj_name=self.vobj_names_mapping)

    def __str__(self):
        return f"VObjFrameOutputNode(props_mapping={self.props_mapping}), \n"\
            f"\tprev={self.prev.__class__.__name__}), \n" \
            f"\tnext={self.next.__class__.__name__})"


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


class Executor:

    def __init__(self, root_plan_node, launch_args):
        self.root_plan_node = root_plan_node
        self._launch_args = add_video_metadata(launch_args)
        self.root_operator = root_plan_node.to_operator(self._launch_args)

    def execute(self):
        while self.root_operator.has_next():
            result = self.root_operator.next()
            yield result
