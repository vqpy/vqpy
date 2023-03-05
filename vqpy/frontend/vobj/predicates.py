from abc import ABC, abstractmethod


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


class And(Predicate):

    def __init__(self, lp: Predicate, rp: Predicate) -> Predicate:
        self.left_pred = lp
        self.right_pred = rp

    def __str__(self):
        return f"And(left={self.left_pred}, right={self.right_pred})"
    
    def get_vobjs(self):
        return self.left_pred.get_vobjs() | self.right_pred.get_vobjs()
    
    def generate_condition_function(self):
        l_f = self.left_pred.generate_condition_function()
        r_f = self.right_pred.generate_condition_function()

        return lambda vobj_data: l_f(vobj_data) and r_f(vobj_data)


class Or(Predicate):

    def __init__(self, lp: Predicate, rp: Predicate) -> Predicate:
        self.left_pred = lp
        self.right_pred = rp

    def get_vobjs(self):
        return self.left_pred.get_vobjs() | self.right_pred.get_vobjs()


class Not(Predicate):

    def __init__(self, p: Predicate) -> Predicate:
        self.pred = p

    def get_vobjs(self):
        return self.pred.get_vobjs()


class IsInstance(Predicate):

    def __init__(self, vobj):
        self.vobj = vobj

    def get_vobjs(self):
        return {self.vobj}

    def __str__(self):
        return f"IsInstance(vobj={self.vobj})"

    def __repr__(self):
        return str(self)

    def generate_condition_function(self):
        return self.vobj.class_name


class Equal(Predicate):

    def __init__(self, left_prop, right_prop):
        self.left_prop = left_prop
        self.right_prop = right_prop

    def __str__(self):
        return f"Equal(left={self.left_prop}, right={self.right_prop})"

    def get_vobjs(self):
        return self.left_prop.get_vobjs() | self.right_prop.get_vobjs()


class GreaterThan(Predicate):

    def __init__(self, left_prop, right_prop):
        self.left_prop = left_prop
        self.right_prop = right_prop

    def __str__(self):
        return f"GreaterThan(left={self.left_prop}, right={self.right_prop})"

    def get_vobjs(self):
        return self.left_prop.get_vobjs() | self.right_prop.get_vobjs()

    def generate_condition_function(self):
        def condition_function(vobj_data: dict):
            if self.left_prop.is_literal():
                l_value = self.left_prop.value
            else:
                l_value = vobj_data[self.left_prop.name]
            if self.right_prop.is_literal():
                r_value = self.right_prop.value
            else:
                r_value = vobj_data[self.right_prop.name]
            return l_value > r_value

        return condition_function