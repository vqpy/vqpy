class InvalidProperty(object):
    """
    A property that is not valid. Includes:
    1. A property that cannot be computed accurately. e.g. image
    2. A property whose dependencies contain invalid properties.
    3.A stateful property whose dependencies do not contain enough
      historical data.
    Note that users who want to define a property only need to handle
    case 1 mannually. Case 2 and 3 are handled by the framework.
    """
    pass


class UnComputedProperty(object):
    pass
