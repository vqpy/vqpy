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
    @vqpy.postproc({'majority': 100})
    def license_plate(self):
        return self.infer('license_plate', {'license_plate': 'openalpr'})

class ListMovingVehicle(vqpy.QueryBase):
    def apply(self, tracks: List[vqpy.VObjBase]) -> List[Dict]:
        return vqpy.vobj_select(
                  vqpy.vobj_filter(tracks, {"__class__": lambda x: x == Vehicle, "bbox_velocity": lambda x: x >= 0.1, "license_plate": lambda x: x is not None}),
                  cond = {"license_plate": lambda x: x}
                )

vqpy.launch(cls_name=COCO_CLASSES,
            cls_type={"car": Vehicle, "truck": Vehicle},
            workers=[ListMovingVehicle()])
