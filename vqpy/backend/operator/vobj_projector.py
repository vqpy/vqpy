from vqpy.backend.operator.base import Operator
from vqpy.backend.frame import Frame
from typing import Callable, Dict, Any
import pandas as pd


class VObjProjector(Operator):
    def __init__(self,
                 prev: Operator,
                 property_name: str,
                 property_func: Callable[[Dict], Any],
                 dependencies: Dict[str, int],
                 class_name: str,
                 filter_index: int = 0,
                 ):
        """
        Filter vobjs based on the condition_func.
        :param prev: previous operator
        :param property_name: the name of the property to be computed.
        :param property_func: a callable function that takes in the dependency
         data and returns the property value. The dependency data is a
         dictionary with the key being the dependency property name and the
         value being a list of the historical data (from old to new:
         [last_n-1th_value, last_n-2th_value, ..., cur_value])
         with length n of history length specified in the dependencies.
        :param dependencies: a dict of the dependencies of the property.
            The key is the name of the dependency property and the value is
            the history length (a non-negative integer) of the dependency
            property. If the value is 0, it means current frame.
        :param class_name: the name of the vobj class to compute the property.
        :param filter_index: the index of the filter.
        """
        self.property_name = property_name
        self.property_func = property_func
        self.dependencies = dependencies
        self.filter_index = filter_index
        self.class_name = class_name
        self._max_history_len = max(dependencies.values())
        columns = ["track_id", "frame_id", "vobj_index"].append(
            list(dependencies.keys()))
        self._hist_buffer = pd.DataFrame(columns=columns)

        super().__init__(prev)

    def _get_cur_frame_dependencies(self, frame):
        # TODO: Add support for video metadata and frame image dependencies
        # 1. get vobj indexes in filter index of class name
        if self.filter_index not in frame.filtered_vobjs:
            raise ValueError("filter_index is not in filtered_vobjs")
        vobj_indexes = frame.filtered_vobjs[self.filter_index][self.class_name]
        # 2. if has track id, get dictionary of key of vobj_index,
        #  value of track id and dependency data
        cur_frame_vobjs = []
        for vobj_index in vobj_indexes:
            vobj_data = frame.vobj_data[self.class_name][vobj_index]
            if "track_id" in vobj_data:
                track_id = vobj_data["track_id"]
                vobj_data_dep = {"vobj_index": vobj_index,
                                 "track_id": track_id,
                                 "frame_id": frame.id}
                assert all([dep_name in vobj_data for dep_name in
                            self.dependencies.keys()]), \
                    f"vobj_data does not have all dependencies. Keys of "
                f"vobj_data: {vobj_data.keys()}. Keys of dependencies: "
                f"{self.dependencies.keys()}"
                vobj_data_dep.update({dep_name: vobj_data[dep_name]
                                      for dep_name in self.dependencies.keys()}
                                     )
                cur_frame_vobjs.append(vobj_data_dep)
        return cur_frame_vobjs

    def _update_hist_buffer(self, cur_frame_vobjs):
        self._hist_buffer.append(cur_frame_vobjs)
        # remove data that older than max history length
        if cur_frame_vobjs:
            cur_frame_id = cur_frame_vobjs[0]["frame_id"]
            # frame_id starts from 0
            oldest_frame_id = cur_frame_id + 1 - self._max_history_len
            if oldest_frame_id >= 0:
                self._hist_buffer = self._hist_buffer[
                    self._hist_buffer["frame_id"] >= oldest_frame_id]

    def _get_hist_dependency(self,
                             dependency_name,
                             track_id,
                             frame_id,
                             hist_len):
        hist_start = frame_id - hist_len
        # return None if there isn't enough history
        if hist_start < 0:
            return None
        # if dependency is the property itself, get history data from
        # hist_start to hist_end-1, and append None to the end as current
        # frame data.
        if dependency_name == self.property_name:
            hist_end = frame_id - 1
        else:
            hist_end = frame_id
        # get dependency data from hist buffer
        row = (self._hist_buffer["track_id"] == track_id) & \
              (self._hist_buffer["frame_id"] >= hist_start) & \
              (self._hist_buffer["frame_id"] <= hist_end)

        # data has been sorted by frame_id (writing order)
        hist_data = self._hist_buffer.loc[row, dependency_name].tolist()
        if dependency_name == self.property_name:
            hist_data = hist_data.append(None)
        return hist_data

    def _compute_property(self, cur_frame_vobjs, frame):
        """
        Compute property for each vobj in cur frame vobjs.
        :param cur_frame_vobjs: a list of vobj data dictionaries, with keys of
        "vobj_index", "track_id", self.dependencies.keys().

        :return: frame with updated property values for filtered and tracked
         vobjs.
        """
        if cur_frame_vobjs:
            assert cur_frame_vobjs[0].keys() == {"vobj_index",
                                                 "track_id",
                                                 "frame_id",
                                                 *self.dependencies.keys()}, \
                "cur_frame_vobjs must have keys of vobj_index, track_id, " \
                "and dependencies.keys()"
        for vobj_data in cur_frame_vobjs:
            vobj_index = vobj_data["vobj_index"]
            track_id = vobj_data["track_id"]
            frame_id = vobj_data["frame_id"]
            dep_data_dict = dict()
            for dependency_name, hist_len in self.dependencies.items():
                # get cur frame dependencies if dependency has history length 0
                if hist_len == 0:
                    dep_data_dict[dependency_name] = vobj_data[dependency_name]
                # get dependency data from hist buffer if dependency data has
                # history length > 0
                else:
                    dep_data = self._get_hist_dependency(dependency_name,
                                                         track_id=track_id,
                                                         frame_id=frame_id,
                                                         hist_len=hist_len)
                    dep_data_dict[dependency_name] = dep_data
            # compute property
            property_value = self.property_func(dep_data_dict)
            # update frame vobj_data with computed property value for
            # corresponding vobj
            frame.vobj_data[self.class_name][vobj_index][self.property_name] =\
                property_value
        return frame

    def next(self) -> Frame:
        if self.prev.has_next():
            frame = self.prev.next()
            cur_frame_vobjs = self._get_cur_frame_dependencies(frame)
            self._update_hist_buffer(cur_frame_vobjs=cur_frame_vobjs)
            frame = self._compute_property(cur_frame_vobjs=cur_frame_vobjs,
                                           frame=frame)
        return frame

# TODO: ADD CrossVobjProjector
