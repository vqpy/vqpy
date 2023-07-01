from vqpy.backend.operator.object_detector import ObjectDetector
from vqpy.backend.plan_nodes.base import AbstractPlanNode


from typing import Optional, Set, Union


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