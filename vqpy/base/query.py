"""The query base class"""

from __future__ import annotations

from typing import List
from ..base.interface import (
    VObjBaseInterface,
    VObjConstraintInterface,
    OutputConfig
)


OUTPUT_FRAME_VOBJ_NUM_NAME = "vobj_num"


class QueryBase(object):
    """the base class of all applied queries"""
    def get_base_setting(self) -> VObjConstraintInterface:
        """Get the setting of this class"""
        ret = self.setting()
        cls: QueryBase = self.__class__.__bases__[0]
        while cls != QueryBase:
            ret = ret + cls.setting()
            cls = cls.__bases__[0]
        return ret

    def vqpy_init(self):
        # The data is initialized here to avoid override of __init__()
        self._query_data = []
        self._setting = self.get_base_setting()
        self._output_configs = self.set_output_configs()

    def vqpy_update(self,
                    frame_id: int,
                    vobjs: List[VObjBaseInterface]):
        data = self._setting.apply(vobjs)
        frame_vobj_num = len(data)
        frame_query_data = dict()
        if self._output_configs.output_frame_vobj_num:
            frame_query_data = {"frame_id": frame_id,
                                OUTPUT_FRAME_VOBJ_NUM_NAME: frame_vobj_num}
        if frame_vobj_num > 0:
            if "frame_id" not in frame_query_data:
                frame_query_data["frame_id"] = frame_id
            frame_query_data["data"] = data
        if frame_query_data:
            self._query_data.append(frame_query_data)

    def vqpy_getdata(self):
        """Returns the query database of the final data"""
        return self._query_data

    def get_setting(self):
        return self._setting

    def get_output_configs(self):
        return self._output_configs

    @staticmethod
    def setting() -> VObjConstraintInterface:
        """
        The database operation setting applied on the per-frame updated tracks.
        return: a VObjConstraint instance containing the required settings.
        """
        raise NotImplementedError

    @staticmethod
    def set_output_configs() -> OutputConfig:
        return OutputConfig()
