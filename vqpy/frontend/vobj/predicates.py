from abc import ABC, abstractmethod
from vqpy.frontend.vobj.common import UnComputedProperty, get_dep_properties


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
        return (
            f"{self.__class__.__name__}\n "
            f"\t(left={self.left_pred}, \n"
            f"\tright={self.right_pred})"
        )

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


class LiteralPredicate(Predicate):
    def __init__(self, left_prop, right_prop):
        self.left_prop = left_prop
        self.right_prop = right_prop

    def __str__(self):
        return (
            f"{self.__class__.__name__}\n"
            f"\t\t(left={self.left_prop},\n"
            f"\t\tright={self.right_prop})"
        )

    def get_vobjs(self):
        return self.left_prop.get_vobjs() | self.right_prop.get_vobjs()

    def get_vobj_properties(self):
        vobj_properties = get_dep_properties(self.left_prop)
        right_vobj_props = get_dep_properties(self.right_prop)
        for prop in right_vobj_props:
            if all([prop.func != vp.func for vp in vobj_properties]):
                vobj_properties.append(prop)
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


class Equal(LiteralPredicate):

    def generate_condition_function(self):
        def condition_function(vobj_data: dict):
            l_value, r_value = self._get_prop_values(vobj_data)
            if isinstance(l_value, UnComputedProperty) or isinstance(
                r_value, UnComputedProperty
            ):
                return False
            return l_value == r_value

        return condition_function


class GreaterThan(LiteralPredicate):

    def generate_condition_function(self):
        def condition_function(vobj_data: dict):
            l_value, r_value = self._get_prop_values(vobj_data)

            if isinstance(l_value, UnComputedProperty) or isinstance(
                r_value, UnComputedProperty
            ):
                return False

            return l_value > r_value

        return condition_function


class Compare(Predicate):

    def __init__(self, prop, compare_func):
        self.prop = prop
        self.compare_func = compare_func

    def __str__(self):
        return f"Compare(prop={self.prop}\n "\
            f"\tcompare_func={self.compare_func.__name__})"

    def get_vobjs(self):
        return self.prop.get_vobjs()

    def get_vobj_properties(self):
        return get_dep_properties(self.prop)

    def _get_prop_value(self, vobj_data):
        if self.prop.is_vobj_property() and self.prop.stateful:
            if self.prop.name in vobj_data:
                return vobj_data[self.prop.name]
            else:
                return UnComputedProperty()
        else:
            return vobj_data[self.prop.name]

    def generate_condition_function(self):
        def condition_function(vobj_data: dict):
            prop_value = self._get_prop_value(vobj_data)

            if isinstance(prop_value, UnComputedProperty):
                return False

            return self.compare_func(prop_value)

        return condition_function
