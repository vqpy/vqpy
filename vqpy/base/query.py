"""The query base class"""

from __future__ import annotations

from typing import List
from ..base.interface import VObjBaseInterface, VObjConstraintInterface


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

    def vqpy_update(self,
                    frame_id: int,
                    vobjs: List[VObjBaseInterface]):
        data = self._setting.apply(vobjs)
        if len(data) > 0:
            self._query_data.append({"frame_id": frame_id, "data": data})

    def vqpy_getdata(self):
        """Returns the query database of the final data"""
        return self._query_data

    def get_setting(self):
        return self._setting

    @staticmethod
    def setting() -> VObjConstraintInterface:
        """
        The database operation setting applied on the per-frame updated tracks.
        return: a VObjConstraint instance containing the required settings.
        """
        raise NotImplementedError
