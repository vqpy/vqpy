from abc import ABC, abstractmethod
from vqpy.frontend.vobj.common import UnComputedProperty


class Predicate(ABC):

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)

    @abstractmethod
    def get_vobjs(self):
        raise NotImplementedError

    @abstractmethod
    def get_vobj_properties(self):
        raise NotImplementedError


class BinaryPredicate(Predicate):

    def __init__(self, lp: Predicate, rp: Predicate) -> Predicate:
        self.left_pred = lp
        self.right_pred = rp

    def __str__(self) -> str:
        return f"{self.__class__.__name__}\n "\
            f"\t(left={self.left_pred}, \n" \
            f"\tright={self.right_pred})"

    def get_vobjs(self):
        return self.left_pred.get_vobjs() | self.right_pred.get_vobjs()

    def get_vobj_properties(self):
        vobj_props = self.left_pred.get_vobj_properties()
        for p in self.right_pred.get_vobj_properties():
            if all([p.func != vp.func for vp in vobj_props]):
                vobj_props.append(p)
        return vobj_props


class And(BinaryPredicate):

    def generate_condition_function(self):
        l_f = self.left_pred.generate_condition_function()
        r_f = self.right_pred.generate_condition_function()

        return lambda vobj_data: l_f(vobj_data) and r_f(vobj_data)


class Or(BinaryPredicate):
    def generate_condition_function(self):
        l_f = self.left_pred.generate_condition_function()
        r_f = self.right_pred.generate_condition_function()

        return lambda vobj_data: l_f(vobj_data) or r_f(vobj_data)


class Not(Predicate):

    def __init__(self, p: Predicate) -> Predicate:
        self.pred = p

    def get_vobjs(self):
        return self.pred.get_vobjs()

    def get_vobj_properties(self):
        return self.pred.get_vobj_properties()


class IsInstance(Predicate):

    def __init__(self, vobj):
        self.vobj = vobj

    def get_vobjs(self):
        return {self.vobj}

    def get_vobj_properties(self):
        return []

    def __str__(self):
        return f"IsInstance(vobj={self.vobj})"

    def __repr__(self):
        return str(self)

    def generate_condition_function(self):
        return self.vobj.class_name


class Literal(Predicate):
    def __init__(self, left_prop, right_prop):
        self.left_prop = left_prop
        self.right_prop = right_prop

    def __str__(self):
        return f"{self.__class__.__name__}\n"\
            f"\t\t(left={self.left_prop},\n" \
            f"\t\tright={self.right_prop})"

    def get_vobjs(self):
        return self.left_prop.get_vobjs() | self.right_prop.get_vobjs()

    def get_vobj_properties(self):
        vobj_properties = []
        if self.left_prop.is_vobj_property():
            vobj_properties.append(self.left_prop)
        # remove duplicates if the same function is used
        if self.right_prop.is_vobj_property():
            if all([self.right_prop.func != vprop.func
                    for vprop in vobj_properties]):
                vobj_properties.append(self.right_prop)
        return vobj_properties

    def _get_prop_values(self, vobj_data):
        def get_value(prop):
            if prop.is_literal():
                return prop.value
            elif prop.is_vobj_property() and prop.stateful:
                if prop.name in vobj_data:
                    return vobj_data[prop.name]
                else:
                    return UnComputedProperty()
            else:
                return vobj_data[prop.name]
        l_value = get_value(self.left_prop)
        r_value = get_value(self.right_prop)
        return l_value, r_value

    @abstractmethod
    def generate_condition_function(self):
        raise NotImplementedError


class Equal(Literal):

    def generate_condition_function(self):
        def condition_function(vobj_data: dict):
            l_value, r_value = self._get_prop_values(vobj_data)
            if isinstance(l_value, UnComputedProperty)\
                    or isinstance(r_value, UnComputedProperty):
                return False
            return l_value == r_value

        return condition_function


class GreaterThan(Literal):

    def generate_condition_function(self):
        def condition_function(vobj_data: dict):
            l_value, r_value = self._get_prop_values(vobj_data)

            if isinstance(l_value, UnComputedProperty)\
                    or isinstance(r_value, UnComputedProperty):
                return False

            return l_value > r_value

        return condition_function

# todo, add `compare` predicate for UDFs
