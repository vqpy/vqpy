from typing import List
from ..base.interface import VObjBaseInterface
from ..database import VObjConstraint

class QueryBase:
    def _begin_query(self):
        self._query_data = []
        general_setting = self.setting()
        cls = self.__class__.__bases__[0]
        while cls != QueryBase:
            general_setting = general_setting + cls.setting()
            cls = cls.__bases__[0]
        self._setting = general_setting
    
    def _update_query(self, frame_id: int, vobjs: List[VObjBaseInterface]):
        self._query_data.append({"frame_id": frame_id, "data": self._setting.apply(vobjs)})
    
    def _end_query(self):
        """Returns the query database of the final data"""
        return self._query_data

    def _get_setting(self):
        return self._setting
    
    @staticmethod
    def setting() -> VObjConstraint:
        """
        The setting for the applied filter/selects on the per-frame updated tracks.
        return: a VObjConstraint instance containing the required settings.
        """
        return VObjConstraint()
