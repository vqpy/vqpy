# Based on demo implementation in Megvii YOLOX repo

import sys
from typing import Dict, List

sys.path.append(".")
sys.path.append("./models/yolo")
sys.path.append("./vqpy_demo")

import vqpy
from vqpy.utils import COCO_CLASSES
from vqpy.video_loader import FrameStream

class Vehicle(vqpy.VObjBase):
    required_fields = ['class_id', 'tlbr']
    def __init__(self, ctx: FrameStream):
        super().__init__(ctx)
        self._list_lp = []
    @vqpy.property()
    def license_plate(self):
        this_lp = vqpy.infer(self, 'license_plate', ['frame', 'tlbr'])
        self._list_lp.append(this_lp)
        if len(self._list_lp) > 100:
            self._list_lp = self._list_lp[1:]
        local_map = {}
        for it in self._list_lp:
            if it in local_map: local_map[it] += 1
            else: local_map[it] = 1
        ret = (None, 0)
        for it, v in local_map.items():
            if it is not None and v > ret[1]:
                ret = (it, v)
        return ret[0]

class ListMovingVehicle(vqpy.QueryBase):
    def apply(self, tracks: List[vqpy.VObjBase]) -> List[Dict]:
        return vqpy.vobj_select(
                  vqpy.vobj_filter(tracks, {"__class__": lambda x: x == Vehicle, "bbox_velocity": lambda x: x >= 0.1, "license_plate": lambda x: x is not None}),
                  cond = {"license_plate": lambda x: x}
                )

vqpy.launch(cls_name=COCO_CLASSES,
            cls_type={"car": Vehicle, "truck": Vehicle},
            workers=[ListMovingVehicle()])
