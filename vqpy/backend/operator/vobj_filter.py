from vqpy.backend.operator.base import Operator
from vqpy.backend.frame import Frame
from typing import Callable, Union, List, Dict


class VObjFilter(Operator):
    def __init__(self,
                 prev: Operator,
                 condition_func: Union[Callable[[Dict], bool], str, List[str]],
                 filter_index: int = 0,
                 ):
        """
        Filter vobjs based on the condition_func.
        :param prev: previous operator
        :param condition_func: a callable function that takes in the data of
            one vobj and returns a bool value.
            If condition_func is a string or a list of string, it should be
            one or more class name.
            The vobjs with the class name(s) will be filtered.
        :param filter_index: the index of the filter.
            If the index is already used, raise ValueError.
        """
        self.condition_func = condition_func
        self.filter_index = filter_index
        super().__init__(prev)

    def _update_filtered_class(self, class_name, frame: Frame):
        if class_name in frame.filtered_vobjs[self.filter_index]:
            raise ValueError(f"filter index {self.filter_index} already \
                contains {class_name}")
        if class_name not in frame.vobj_data:
            frame.filtered_vobjs[self.filter_index][class_name] = []
        else:
            frame.filtered_vobjs[self.filter_index][class_name] = \
                list(range(len(frame.vobj_data[class_name])))
        return frame

    def _update_filtered_vobjs(self, frame: Frame):
        if isinstance(self.condition_func, str):
            frame = self._update_filtered_class(self.condition_func, frame)
        elif isinstance(self.condition_func, list):
            assert all(isinstance(cls_name, str)
                       for cls_name in self.condition_func), \
                "condition_func must be either a string or a list of string"
            class_names = self.condition_func
            for class_name in class_names:
                frame = self._update_filtered_class(class_name, frame)
        else:
            if not callable(self.condition_func):
                raise ValueError("condition_func must be either a string \
                    or a function")
            assert self.filter_index not in frame.filtered_vobjs, \
                "Filter on properties before filtering on vobj classes."
            filtered_vobjs = frame.filtered_vobjs[self.filter_index]
            for class_name, vobj_indexes in filtered_vobjs.items():
                new_vobj_indexes = []
                for index in vobj_indexes:
                    vobj_data = frame.vobj_data[class_name][index]
                    assert self.property_name in vobj_data, \
                        f"property_name {self.property_name} of index {index} \
                        of {class_name} is not computed before filtering."
                    if self.condition_func(vobj_data):
                        new_vobj_indexes.append(index)
                # update filtered vobjs on frame
                frame.filtered_vobjs[self.filter_index][class_name] = \
                    new_vobj_indexes

        return frame

    def next(self) -> Frame:
        if self.has_next():
            frame = self.prev.next()
            frame = self._update_filtered_vobjs(frame)
            return frame
        else:
            raise StopIteration


class VObjPropertyFilter(VObjFilter):
    def __init__(self,
                 prev: Operator,
                 property_name: str,
                 property_condition_func: Callable[[Dict], bool],
                 filter_index: int = 0,
                 ):
        """
        Filter vobjs based on property value.
        :param prev: previous operator
        :param property_condition_func: a callable function that takes in a 
            vobj property value and returns a bool value.
        :filter_index: the index of the filter.
        :param property_name: the name of the property on vobj.
        """
        def condition_func(vobj_data: Dict):
            assert property_name in vobj_data, \
                        f"property_name {self.property_name} is not computed \
                            before filtering."
            return property_condition_func(vobj_data[property_name])
        super().__init__(prev, condition_func, filter_index)
