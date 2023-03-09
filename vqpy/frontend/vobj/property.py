from vqpy.frontend.vobj.predicates import Equal, GreaterThan
from typing import Dict, Callable
from abc import ABC


class Property(ABC):

    def get_vobjs(self):
        return set()

    def __eq__(self, other):

        if not isinstance(other, Property):
            other = Literal(other)

        return Equal(self, other)

    def __gt__(self, other):
        if not isinstance(other, Property):
            other = Literal(other)
        return GreaterThan(self, other)

    def __lt__(self, other):
        if not isinstance(other, Property):
            other = Literal(other)
        return GreaterThan(other, self)

    def is_literal(self):
        return False

    def is_vobj_property(self):
        return False


class BuiltInProperty(Property):

    def __init__(self, vobj, name: str) -> None:
        self.vobj = vobj
        self.name = name
        super().__init__()

    def get_vobjs(self):
        return {self.vobj}


class Literal(Property):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Literal(value={self.value})"

    def is_literal(self):
        return True


class VobjProperty(Property):

    def __init__(self, vobj, inputs: Dict[str, int], func: Callable):
        self.vobj = vobj
        self.inputs = inputs
        self.func = func
        self.name = func.__name__
        self.stateful = any([hist_len > 0 for hist_len in inputs.values()])

    def __call__(self, *args, **kwargs):
        return self.func(self.vobj, *args, **kwargs)

    def __str__(self):
        return f"VObjProp(vobj={self.vobj.__class__.__name__},\
             inputs={self.inputs}, Prop={self.name})"

    def get_vobjs(self):
        return {self.vobj}

    def is_vobj_property(self):
        return True
