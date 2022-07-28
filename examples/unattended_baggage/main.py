from typing import List
import vqpy
from __future__ import annotations
import numpy as np

from vqpy.utils import COCO_SUPERCLASSES

def distance(person : Person, accessory : Accessory):
    d = person.getv("coordinate", -1) - accessory.getv("coordinate", -1)
    return float(np.sqrt(sum(d * d)))

# TODO: add required input_fields and output_fields

class Person(vqpy.VObjBase):
    pass

class Accessory(vqpy.VObjBase):
    threshold = 10
    OwnDist = 5
    @vqpy.property()
    @vqpy.stateful(2)
    @vqpy.worker(({"type": Person, "is_activated": True}))
    def owner(self, person : List[vqpy.VObjBase]):
        last_owner = self.getv("owner", -2)
        if last_owner is not None:
            p = vqpy.vobj_filter(person, {"track_id": lambda x: x == last_owner})
            if len(p) == 1 and distance(p[0], self) < Accessory.OwnDist:
                return p[0]
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
        objs: List[Accessory] = vqpy.vobj_select(tracks, {"__class__": lambda x: x == Accessory, "abandoned": lambda x: x})
        return objs.getv("image")

vqpy.launch(cls_type = COCO_SUPERCLASSES,
            cls_dict = {"person": Person, "accessory": Accessory},
            workers = [TriggerAlarm()])
