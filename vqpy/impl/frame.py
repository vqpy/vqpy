from typing import Dict, List
from ..base.interface import VObjGeneratorType, VObjBaseInterface
from collections import defaultdict
from ..utils.video import FrameStream


class Frame:
    def __init__(self, ctx: FrameStream):
        self.ctx = ctx
        # {vobj_type: {vobj_id: vobj_instance}}
        self.vobjs: Dict[VObjGeneratorType, Dict[int, VObjBaseInterface]] = \
            defaultdict(dict)
        self.tracked_vobj_ids: Dict[VObjGeneratorType, set(int)] = \
            defaultdict(set)
        self.lost_vobj_ids: Dict[VObjGeneratorType, set(int)] = \
            defaultdict(set)

    def set_vobjs(self, vobjs):
        self.vobjs = vobjs

    def update_vobjs(self,
                     vobj_type: VObjGeneratorType,
                     track_id: int,
                     data: Dict,
                     ):
        # create new vobj if doesn't exist
        if track_id not in self.vobjs[vobj_type]:
            # each vobj has reference to Frame
            # temporary, should be querying on Frame
            new_vobj = vobj_type(self)
            self.vobjs[vobj_type][track_id] = new_vobj
        else:
            # update _frame if vobj already exists
            self.vobjs[vobj_type][track_id]._frame = self
        # update vobj
        vobj = self.vobjs[vobj_type][track_id]
        vobj.update(data)

        # update tracked and lost vobjs
        if data:
            self.tracked_vobj_ids[vobj_type].add(track_id)
            assert "tlbr" in self.vobjs[vobj_type][track_id]._datas[-1]

        else:
            self.lost_vobj_ids[vobj_type].add(track_id)

    def get_tracked_vobjs(self,
                          vobj_type: VObjGeneratorType,
                          ) -> List[VObjBaseInterface]:
        ids = self.tracked_vobj_ids[vobj_type]
        id_vobjs = self.vobjs[vobj_type]
        tracked_vobjs = [id_vobjs[id] for id in ids]
        return tracked_vobjs

    def get_lost_vobjs(self,
                       vobj_type: VObjGeneratorType,
                       ) -> List[VObjBaseInterface]:
        ids = self.lost_vobj_ids[vobj_type]
        id_vobjs = self.vobjs[vobj_type]
        lost_vobjs = [id_vobjs[id] for id in ids]
        return lost_vobjs
