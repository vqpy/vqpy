from vqpy.backend.operator.video_reader import VideoReader
from vqpy.backend.operator.object_detector import ObjectDetector
from vqpy.backend.operator.vobj_filter import VObjFilter
from vqpy.backend.operator.frame_filter import VObjFrameFilter
from vqpy.backend.operator.tracker import Tracker
from vqpy.backend.operator.vobj_projector import VObjProjector
from vqpy.frontend.query import QueryBase
from vqpy.frontend.vobj import VObjBase
from vqpy.frontend.vobj.predicates import Predicate, IsInstance
from abc import abstractmethod
from vqpy.frontend.vobj.property import VobjProperty
from typing import Set, Union, Optional, Dict, Callable, Any, List


class AbstractPlanNode:

    def __init__(self):
        self.next = None
        self.prev = None

    def set_prev(self, plan_node):
        '''
        Set the plan_node as the current node's child node. The current node will consume or depends on the child node.

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
        return f"PlanNode({self.__class__.__name__}, prev={self.prev.__class__.__name__}, next={self.next.__class__.__name__})"


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
        return f"ProjectorNode(class_name={self.class_name}, " \
                f"property_name={self.projection_field.field_name}, " \
                f"filter_index={self.filter_index}), " \
                f"dependencies={self.projection_field.dependent_fields})," \
                f"prev={self.prev.__class__.__name__}), "\
                f"next={self.next.__class__.__name__})"


class TrackerNode(AbstractPlanNode):
    
    def __init__(self,
                 class_name: str,
                 filter_index: Optional[int] = None,
                 tracker_name: str = "byte",
                 **tracker_kwargs):
        self.class_name = class_name
        self.filter_index = filter_index
        self.tracker_name = tracker_name
        self.tracker_kwargs = tracker_kwargs
        super().__init__()

    def to_operator(self, launch_args: dict):
        return Tracker(
            prev=self.prev.to_operator(launch_args),
            class_name=self.class_name,
            filter_index=self.filter_index,
            tracker_name=self.tracker_name,
            **self.tracker_kwargs
        )

    def __str__(self):
        return f"TrackerNode(class_name={self.class_name}, " \
            f"filter_index={self.filter_index}, " \
            f"tracker_name={self.tracker_name}), " \
            f"prev={self.prev.__class__.__name__}), "\
            f"next={self.next.__class__.__name__})"


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
        return f"VObjFilterNode(predicate={self.predicate}, " \
               f"filter_index={self.filter_index}), " \
               f"prev={self.prev.__class__.__name__}), "\
               f"next={self.next.__class__.__name__})"


class VObjFrameFilterNode(AbstractPlanNode):

    def __init__(self, filter_index: int = 0):
        self.filter_index = filter_index
        super().__init__()

    def to_operator(self, lauch_args: dict):
        return VObjFrameFilter(prev=self.prev.to_operator(lauch_args),
                               vobj_filter_index=self.filter_index)

    def __str__(self):
        return f"VObjFrameFilterNode(filter_index={self.filter_index}), prev={self.prev.__class__.__name__}), next={self.next.__class__.__name__})"


class Planner:

    def print_plan(self, node: AbstractPlanNode = None):
        print(node)
        if node.get_prev() is not None:
            self.print_plan(node.get_prev())

    def parse(self, query_obj: QueryBase):
        input_node = VideoReaderNode()
        output_node = self._create_object_detector_node(query_obj, input_node)
        output_node = self._create_tracker_node(query_obj, output_node)
        output_node = self._create_vobj_class_filter_node(query_obj, output_node)
        output_node = self._create_pre_filter_projector(query_obj, output_node)
        output_node = self._create_frame_filter_node(query_obj, output_node)
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
            TrackerNode(class_name=class_name,
                        fps=24.0)
        )

    def _create_pre_filter_projector(self, query_obj: QueryBase, input_node):
        frame_constraints = query_obj.frame_constraint()
        node = input_node

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

        return node

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


class Executor:

    def __init__(self, root_plan_node, lanuch_args):
        self.root_plan_node = root_plan_node
        self.root_operator = root_plan_node.to_operator(lanuch_args)

    def execute(self):
        result = []
        while self.root_operator.has_next():
            frame = self.root_operator.next()
            result.append(frame)
        return result
