import sys

sys.path.append(".")            # enable us to import vqpy module

from typing import List

import vqpy


class Vehicle(vqpy.VObjBase):
    required_fields = ['class_id', 'tlbr']
    @vqpy.property()
    @vqpy.postproc({'majority': 100})
    def license_plate(self):
        return self.infer('license_plate', {'license_plate': 'openalpr'})

class ListMovingVehicle(vqpy.QueryBase):
    def __init__(self):
        self.n_frames = 0
        self.n_movings = 0
        self.set_movings = {}
    
    def apply(self, tracks: List[vqpy.VObjBase]) -> List[vqpy.VObjBase]:
        movings = vqpy.vobj_select(
                    vqpy.vobj_filter(tracks,
                    {'__class__': lambda x: x == Vehicle,
                     'bbox_velocity': lambda x: x >= 0.1,
                     'license_plate': lambda x: x is not None}),
                    cond = {'track_id': lambda x: x,
                            'license_plate': lambda x: x}
                  )
        self.n_frames += 1
        self.n_movings += len(movings)
        for moving in movings:
            self.set_movings[moving['track_id']] = moving['license_plate']
    
    def finalize(self) -> None:
        print('Average moving vehicle count = {%.3f}' % (self.n_movings / self.n_frames))
        print('All moving vehicle count = {%d}' % len(self.set_movings.items()))
        for item in self.set_movings.items(): print(item)

worker = ListMovingVehicle()

vqpy.launch(cls_name=vqpy.COCO_CLASSES,
            cls_type={"car": Vehicle, "truck": Vehicle},
            workers=[worker])

worker.finalize()
