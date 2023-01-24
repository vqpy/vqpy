from vqpy.property_lib.wrappers import vqpy_func_logger


@vqpy_func_logger(['image'], ['license_plate'], [], required_length=1)
def license_plate_openalpr(obj, image):
    """recognize license plate using OpenAlpr"""
    from vqpy.property_lib.vehicle.models.openalpr import GetLP
    return [GetLP(image)]


@vqpy_func_logger(['image'], ['license_plate'], [], required_length=1)
def license_plate_lprnet(obj, image):
    """recognize license plate using LPRNet"""
    from vqpy.property_lib.vehicle.models.lprnet import GetLP
    return [GetLP(image)]
