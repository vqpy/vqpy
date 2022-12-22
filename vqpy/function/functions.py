"""VQPy basic libfunctions"""

from .logger import vqpy_func_logger


@vqpy_func_logger(['tlbr'], ['bbox_velocity'], ['tlbr'], required_length=2)
def bbox_velocity(obj, tlbr):
    """compute the bounding box velocity using the center
    estimate the scaling by the height of box
    """
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
    """crop the image of object from bounding box"""
    from vqpy.utils.images import crop_image
    return [crop_image(frame, tlbr)]


@vqpy_func_logger(['image'], ['license_plate'], [], required_length=1)
def license_plate_lprnet(obj, image):
    """recognize license plate using LPRNet"""
    from vqpy.models.lprnet import GetLP
    return [GetLP(image)]


@vqpy_func_logger(['image'], ['license_plate'], [], required_length=1)
def license_plate_openalpr(obj, image):
    """recognize license plate using OpenAlpr"""
    from vqpy.models.openalpr import GetLP
    return [GetLP(image)]


@vqpy_func_logger(['tlbr'], ['coordinate'], [], required_length=1)
def coordinate_center(obj, tlbr):
    """compute the center of the bounding box"""
    return [(tlbr[:2] + tlbr[2:]) / 2]


@vqpy_func_logger(['tlbr'], ['bottom_center'], [], required_length=1)
def bottom_center_coordinate(obj, tlbr):
    """compute the coordinate of bottom center of the bounding box"""
    x = (tlbr[0] + tlbr[2]) / 2
    y = tlbr[3]
    return [(x, y)]
