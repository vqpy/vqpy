from typing import Any, Callable, Dict
from vqpy.backend.operator.vobj_projector import VObjProjector
from vqpy.backend.plan_nodes.base import AbstractPlanNode


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