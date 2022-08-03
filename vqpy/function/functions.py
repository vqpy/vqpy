from .logger import vqpy_func_logger

@vqpy_func_logger(['tlbr'], ['bbox_velocity'], ['tlbr'], required_length=2)
def bbox_velocity(obj, tlbr):
    from math import sqrt
    tlbr_c, tlbr_p = tlbr, obj.getv('tlbr', -2)
    center_c = (tlbr_c[:2] + tlbr_c[2:]) / 2
    center_p = (tlbr_p[:2] + tlbr_p[2:]) / 2
    tlbr_avg = (tlbr_c + tlbr_p) / 2
    scale = (tlbr_avg[3] - tlbr_avg[1]) / 1.5
    dcenter = (center_c - center_p) / scale * obj._ctx.fps
    v = sqrt(sum(dcenter * dcenter))
    return [v]

@vqpy_func_logger(['frame', 'tlbr'], ['image'], [], required_length=1)
def image_boundarycrop(obj, frame, tlbr):
    from vqpy.utils.images import CropImage
    return [CropImage(frame, tlbr)]

@vqpy_func_logger(['image'], ['license_plate'], [], required_length=1)
def license_plate_lprnet(obj, image):
    from vqpy.models.lprnet import GetLP
    return [GetLP(image)]

@vqpy_func_logger(['image'], ['license_plate'], [], required_length=1)
def license_plate_openalpr(obj, image):
    from vqpy.models.openalpr import GetLP
    return [GetLP(image)]

@vqpy_func_logger(['tlbr'], ['coordinate'], [], required_length=1)
def coordinate_center(obj, tlbr):
    return (tlbr[:2] + tlbr[2:]) / 2
