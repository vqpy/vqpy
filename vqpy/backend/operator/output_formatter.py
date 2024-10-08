from vqpy.backend.operator.base import Operator
from vqpy.backend.frame import Frame
from typing import Dict, List


class FrameOutputFormatter(Operator):
    def __init__(
        self,
        prev: Operator,
        filter_index_to_class_name_to_property_names: Dict[
            int, Dict[str, List[str]]
        ],
        filter_index_to_vobj_name: Dict[int, str],
        other_frame_fields: List[str] = None,
    ):
        """
        Output Formatter for the frame_output function in the QueryBase class.
        For example, if the frame_output function is:
        def frame_output(self):
            return (self.person.center,
                    self.car1.bbox, self.car1.velocity,
                    self.car2.bbox)
        Then filter_index_to_class_name_to_property_names should be:
            {0: {"person": ["center"]},
             1: {"car": ["bbox", "velocity"]},
             2: {"car": ["bbox"]}}
        And filter_index_to_vobj_name should be:
            {0: "person",
             1: "car1",
             2: "car2"}
        """
        self.prev = prev
        self.properties_mapping = filter_index_to_class_name_to_property_names
        self.vobj_names_mapping = filter_index_to_vobj_name
        self.other_frame_fields = FrameOutputFormatter._check_input(
            other_frame_fields
        )

    @staticmethod
    def _check_input(other_frame_fields):
        if other_frame_fields is None:
            other_frame_fields = []
        elif isinstance(other_frame_fields, str):
            other_frame_fields = [other_frame_fields]
        else:
            if not isinstance(other_frame_fields, list):
                raise TypeError(
                    "other_frame_fields must be a list of strings."
                )
            if not isinstance(other_frame_fields[0], str):
                raise TypeError(
                    "other_frame_fields must be a list of strings."
                )
            other_frame_fields = other_frame_fields
        return other_frame_fields

    def next(self) -> Frame:
        """
        Returns: A dictionary with frame id as well as the properties
         specified in the frame_output function (prefixed with vobj name).
            {
                "frame_id": int,
                "person": [{"center": float}, ...],
                "car1": [{"bbox": float, "velocity": float}, ...],
                "car2": [{"bbox": float}, ...]
                other_frame_fields: ...
            }
        """
        output = dict()
        if self.prev.has_next():
            frame = self.prev.next()
            output["frame_id"] = frame.id
            for field in self.other_frame_fields:
                if field not in frame.kwargs:
                    raise ValueError(f"Field {field} not in frame args.")
                output[field] = frame.kwargs[field]
            for filter_index, class2props in self.properties_mapping.items():
                vobj_name = self.vobj_names_mapping[filter_index]
                output[vobj_name] = []
                for class_name, property_names in class2props.items():
                    filt_vobjs = frame.filtered_vobjs[filter_index][class_name]
                    for vobj_index in filt_vobjs:
                        vobj = frame.vobj_data[class_name][vobj_index]
                        vobj_dict = dict()
                        for property_name in property_names:
                            vobj_dict[property_name] = vobj[property_name]
                        output[vobj_name].append(vobj_dict)
        return output
