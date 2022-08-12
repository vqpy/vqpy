import sys

sys.path.append('.')

import numpy as np

import vqpy

class Vehicle(vqpy.VObjBase):
    required_fields = ['class_id', 'tlbr']
    @vqpy.property()
    @vqpy.postproc({'majority': 100})
    def license_plate(self):
        return self.infer('license_plate', {'license_plate': 'openalpr'})

class ListMovingVehicle(vqpy.QueryBase):
    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        filter_cons = {'__class__': lambda x: x == Vehicle,
                       'bbox_velocity': lambda x: x >= 0.1}
        select_cons = {'track_id': None,
                       'license_plate': None}
        return vqpy.VObjConstraint(filter_cons=filter_cons, select_cons=select_cons, filename='moving')

# add general built-in function to library
@vqpy.vqpy_func_logger(['image'], ['dominant_color'], [], required_length=1)
def extract_color(obj, image: np.ndarray):
    # scratchy implementation
    datas = {}
    nrows, ncols, _ = image.shape
    for row in range(nrows):
        for col in range(ncols):
            v = np.round(image[row, col] * 255.0) / 32          # 8x8x8 color grid
            v = tuple(int(x) for x in v)
            if v not in datas: datas[v] = 1
            else: datas[v] += 1
    color, best = None, 0
    for nc, value in datas.items():
        if value > best:
            color = np.array(nc) * 32 / 255.0
            best = value
    return [color]

class ListRedMovingVehicle(ListMovingVehicle):
    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        filter_cons = {'dominant_color': lambda x: np.dot(x / np.linalg.norm(x), np.array([1, 0, 0])) > 0.5}
        return vqpy.VObjConstraint(filter_cons=filter_cons, filename="redmoving")

vqpy.launch(cls_name=vqpy.COCO_CLASSES,
            cls_type={"car": Vehicle, "truck": Vehicle},
            workers=[ListRedMovingVehicle()])
