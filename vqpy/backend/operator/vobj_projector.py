from vqpy.backend.operator.base import Operator
from vqpy.backend.frame import Frame
from typing import Callable, Dict, Any
import pandas as pd
import numpy as np
from vqpy.utils.images import crop_image
from vqpy.common import InvalidProperty

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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
            property. If the value is 0, it means current frame. If the value
            is 1, it means the last frame and the current frame.
        :param class_name: the name of the vobj class to compute the property.
        :param filter_index: the index of the filter.
        """
        self.property_name = property_name
        self.property_func = property_func
        self.dependencies = dependencies
        self.filter_index = filter_index
        self.class_name = class_name
        self._hist_dependencies = {name: hist_len
                                   for name, hist_len in
                                   self.dependencies.items()
                                   if hist_len > 0}
        self._non_hist_dependencies = {name: hist_len
                                       for name, hist_len in
                                       self.dependencies.items()
                                       if hist_len == 0}
        self._self_dep = self.property_name in self._hist_dependencies
        self._stateful = len(self._hist_dependencies) > 0
        self._max_hist_len = max(dependencies.values())
        columns = ["track_id", "frame_id", "vobj_index"] + \
            list(self._hist_dependencies.keys())
        self._hist_buffer = pd.DataFrame(columns=columns)

        super().__init__(prev)

    def _get_cur_frame_dependencies(self, frame):
        # TODO: Add support for video metadata and frame image dependencies
        # 1. get vobj indexes in filter index of class name
        if self.filter_index not in frame.filtered_vobjs:
            raise ValueError("filter_index is not in filtered_vobjs")
        vobj_indexes = frame.filtered_vobjs[self.filter_index][self.class_name]
        # 2. get dependencies that need to be saved as history and current
        # frame dependencies.
        hist_deps = []
        non_hist_deps = []
        for vobj_index in vobj_indexes:
            vobj_data = frame.vobj_data[self.class_name][vobj_index].copy()
            if self._stateful and "track_id" not in vobj_data:
                continue
            else:
                # dependency of "image", which is the frame image cropped with
                #  the vobj's bbox
                if "image" in self.dependencies:
                    assert "tlbr" in vobj_data, "vobj_data does not have tlbr."
                    vobj_image = crop_image(frame.image, vobj_data["tlbr"])
                    vobj_data["image"] = vobj_image

                # dependency in video metadata
                # including "frame_width", "frame_height", "fps", "n_frames"
                for key in frame.video_metadata:
                    if key in self.dependencies:
                        vobj_data[key] = frame.video_metadata[key]

                # sanity check: dependencies should be in vobj_data
                assert all([dep_name in vobj_data
                            for dep_name in self.dependencies.keys()
                            if not dep_name == self.property_name]), \
                    "vobj_data does not have all dependencies. Keys of "
                f"vobj_data: {vobj_data.keys()}. Keys of dependencies: "
                f"{self.dependencies.keys()}"
                # dependency data as current frame dependency
                cur_dep = {dep_name: vobj_data[dep_name]
                           for dep_name in self._non_hist_dependencies.keys()}
                cur_dep.update(
                    {"vobj_index": vobj_index})

                non_hist_deps.append(cur_dep)

                # dependency data to be saved as history
                if self._stateful:
                    hist_dep = {dep_name: vobj_data[dep_name]
                                for dep_name in self._hist_dependencies.keys()
                                if not dep_name == self.property_name}
                    hist_dep.update(
                        {"vobj_index": vobj_index,
                            "track_id": vobj_data["track_id"],
                            "frame_id": frame.id})

                    hist_deps.append(hist_dep)
        # sanity check
        if not self._stateful:
            assert not hist_deps, "stateful_deps should be empty"
        return non_hist_deps, hist_deps

    def _update_hist_buffer(self, hist_deps):
        # self._hist_buffer = self._hist_buffer.append(hist_deps)
        self._hist_buffer = pd.concat([self._hist_buffer,
                                       pd.DataFrame.from_dict(hist_deps)])

        # remove data that older than max history length
        cur_frame_id = hist_deps[0]["frame_id"]
        # frame_id starts from 0
        oldest_frame_id = cur_frame_id + 1 - (self._max_hist_len + 1)
        if oldest_frame_id >= 0:
            self._hist_buffer = self._hist_buffer[
                self._hist_buffer["frame_id"] >= oldest_frame_id]

    def _get_hist_dependency(self,
                             dependency_name,
                             track_id,
                             frame_id,
                             hist_len):
        # todo: allow user to fill missing data with a default value
        # currently fill with None
        hist_start = frame_id - hist_len
        # return None if there isn't enough history
        if hist_start < 0:
            return None, False

        hist_end = frame_id - 1
        # get dependency data from hist buffer
        row = (self._hist_buffer["track_id"] == track_id) & \
              (self._hist_buffer["frame_id"] >= hist_start) & \
              (self._hist_buffer["frame_id"] <= hist_end)
        hist_df = self._hist_buffer.loc[row, ["frame_id", dependency_name]].\
            set_index("frame_id").reindex(range(hist_start, hist_end + 1))
        # fill missing frames with None
        hist_df = hist_df.replace(np.nan, None)
        hist_data = hist_df[dependency_name].tolist()

        # hist_data contains history data
        assert len(hist_data) == hist_len

        return hist_data, True

    def _compute_property(self, non_hist_data, hist_data, frame):
        # Todo: allow user to fill property without enough history with a
        # default value. Currently fill with None
        output_hist_data = hist_data.copy()
        for i, cur_dep in enumerate(non_hist_data):
            vobj_index = cur_dep["vobj_index"]

            dep_data_dict = dict()
            all_enough = True
            all_valid = True
            for dependency_name, hist_len in self._hist_dependencies.items():
                hist_dep = output_hist_data[i]
                assert hist_dep["vobj_index"] == vobj_index
                if dependency_name != self.property_name:
                    assert dependency_name in hist_dep, \
                        f"dependency {dependency_name} is not in hist_dep"
                track_id = hist_dep["track_id"]
                frame_id = hist_dep["frame_id"]
                dep_data, enough = self._get_hist_dependency(dependency_name,
                                                             track_id=track_id,
                                                             frame_id=frame_id,
                                                             hist_len=hist_len)
                if enough:
                    # add current frame dependency data
                    if dependency_name != self.property_name:
                        dep_data.append(hist_dep[dependency_name])
                    else:
                        dep_data.append(None)
                    assert len(dep_data) == hist_len + 1
                    valid = all([not isinstance(d, InvalidProperty)
                                 for d in dep_data])
                    all_valid = all_valid and valid
                all_enough = all_enough and enough

                dep_data_dict[dependency_name] = dep_data

            for dependency_name in self._non_hist_dependencies:
                assert dependency_name in cur_dep, \
                    f"dependency {dependency_name} is not in cur_dep"
                dep_data = cur_dep[dependency_name]
                valid = not isinstance(dep_data, InvalidProperty)
                all_valid = all_valid and valid
                dep_data_dict[dependency_name] = dep_data

            # compute property only when there is enough history and all
            # dependency data are valid
            if all_enough and all_valid:
                property_value = self.property_func(dep_data_dict)
            elif self._self_dep:
                # if not enough history or invalid data, set property that
                # depends on itself to None to avoid infinite loop
                property_value = None
            else:
                # if not enough history or invalid data, set property to
                # InvalidProperty
                property_value = InvalidProperty()

            # update frame vobj_data with computed property value for
            # corresponding vobj
            vobj_data = frame.vobj_data[self.class_name][vobj_index]
            vobj_data[self.property_name] = property_value

            # update output_hist_data with computed self dependent property
            # value
            if self._self_dep:
                hist_dep[self.property_name] = property_value

        return frame, output_hist_data

    def next(self) -> Frame:
        if self.prev.has_next():
            frame = self.prev.next()
            non_hist_data, hist_data = self._get_cur_frame_dependencies(frame)
            frame, output_hist_data = self._compute_property(non_hist_data,
                                                             hist_data,
                                                             frame=frame)
            if self._stateful and hist_data:
                self._update_hist_buffer(hist_deps=output_hist_data)
        return frame

# TODO: ADD CrossVobjProjector
