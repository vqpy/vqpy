from typing import Dict, Callable
from vqpy.frontend.vobj.property import BuiltInProperty, VobjProperty
from abc import ABC


class VObjBase(ABC):

    def __init__(self, name: str = None):
        self.tlbr = BuiltInProperty(self, "tlbr")
        self.score = BuiltInProperty(self, "score")
        self.cls = BuiltInProperty(self, "cls")
        self.image = BuiltInProperty(self, "image")
        self.fps = BuiltInProperty(self, "fps")
        self.frame_width = BuiltInProperty(self, "frame_width")
        self.frame_height = BuiltInProperty(self, "frame_height")
        self.n_frames = BuiltInProperty(self, "n_frames")
        self.name = name or self.__class__.__name__

    def get_builtin_property_names(self):
        return {p for p in dir(self)
                if isinstance(getattr(self, p), BuiltInProperty)}

    def get_property(self, name):
        return getattr(self, name)


def vobj_property(inputs: Dict[str, int]):

    def decorator(func: Callable):
        def create_vobj_property(self):
            return VobjProperty(self, inputs, func)
        return property(create_vobj_property)

    return decorator


class MyVobj(VObjBase):

    @vobj_property(inputs={})
    def add_one(self, x):
        return 1 + x

    @vobj_property(inputs={})
    def add_two(self, x):
        return 2 + x


if __name__ == "__main__":

    obj = MyVobj()
    print((obj.add_one == 0.5) & (obj.add_two == 2))
    print(MyVobj().add_one(2))
