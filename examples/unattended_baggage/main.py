"""
from __future__ import annotations
import sys

sys.path.append(".")
sys.path.append("./models/yolo")
sys.path.append("./vqpy_demo")

from typing import List
import vqpy
import numpy as np

from vqpy.utils import COCO_SUPERCLASSES

def distance(person : Person, accessory : Accessory):
    d = person.getv("coordinate") - accessory.getv("coordinate")
    return float(np.sqrt(sum(d * d)))

class Person(vqpy.VObjBase):
    pass

class Accessory(vqpy.VObjBase):
    required_fields = [""]
    threshold = 10
    OwnDist = 5
    @vqpy.property()
    @vqpy.stateful(2)
    @vqpy.access_data(({"__class__": lambda x: x == Person}))
    def owner(self, person : List[Person]):
        # TODO: Should we allow the accessory maintains a historic version of value?
        last_owner = self.getv("owner", -2)
        if last_owner is not None:
            p: List[Person] = vqpy.vobj_filter(person, {"track_id": lambda x: x == last_owner})
            if len(p) == 1 and distance(p[0], self) < Accessory.OwnDist:
                return p[0].getv("track_id")
        p: Person = vqpy.vobj_argmin(person, distance, (None, self))
        if p is not None and distance(p, self) < Accessory.OwnDist:
            return p
        return None
    @vqpy.property()
    @vqpy.stateful(2)
    def lasting_time(self):
        last_value = self.getv("lasting_time", -2)
        if last_value is None: last_value = 0
        return last_value + 1.0 / self.getv("fps")
    @vqpy.property()
    def abandoned(self):
        return self.getv("lasting_time") > Accessory.threshold

class TriggerAlarm(vqpy.QueryBase):
    def apply(self, tracks : List[vqpy.VObjBase]):
        return vqpy.vobj_select(vqpy.vobj_filter(tracks,
                 {"__class__": lambda x: x == Accessory, "abandoned": lambda x: x}), "image")

vqpy.launch(cls_name = COCO_SUPERCLASSES,
            cls_type = {"person": Person, "accessory": Accessory},
            workers = [TriggerAlarm()])
"""
