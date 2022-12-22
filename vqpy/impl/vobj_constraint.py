from __future__ import annotations

from typing import Callable, Dict, List, Optional
from ..base.interface import VObjBaseInterface, VObjConstraintInterface
from ..utils.filters import continuing


class VObjConstraint(VObjConstraintInterface):
    """The constraint on VObj instances, helpful when applying queries"""

    def __init__(
        self,
        filter_cons: Dict[str, Optional[Callable]] = {},
        select_cons: Dict[str, Optional[Callable]] = {},
        filename: str = "data",
    ):
        """Initialize a VObj constraint instances

        filter_cons (Dict[str, Optional[Callable]], optional):
            the filter constraints, an item pair (key, cond) denotes the
        property 'key' of the VObj instance should satisfies cond(key) is
        True. When cond is None, the instance should satisfies key is True.
        Defaults to {}.

        select_cons (Dict[str, Optional[Callable]], optional):
            the select constraints, an item pair (key, proc) denotes select
        property 'key' of the VObj instance and then apply 'proc' on it. When
        cond is None, we do the identity transformation. Defaults to {}.

        filename (str, optional): the saved json name. Defaults to "data".
        """
        self.filter_cons = {key: (lambda x: x is True) if func is None
                            else func for (key, func) in filter_cons.items()}
        self.select_cons = {key: (lambda x: x) if func is None
                            else func for (key, func) in select_cons.items()}
        self.filename = filename

    def __add__(self, other: VObjConstraint) -> VObjConstraint:
        """merge constraints in the form subclass + superclass"""
        filter_cons = self.filter_cons.copy()
        # merge filter constraints
        for key, cond in other.filter_cons.items():
            if key in filter_cons:
                filter_cons[key] = lambda x: filter_cons[key](x) and cond(x)
            else:
                filter_cons[key] = cond
        # always use select_cons in derived class
        select_cons = self.select_cons.copy()
        return VObjConstraint(filter_cons, select_cons, self.filename)

    def filter(self, objs: List[VObjBaseInterface]) -> List[VObjBaseInterface]:
        """filter the list of vobjects from the constraint"""
        ret: List[VObjBaseInterface] = []
        for obj in objs:
            ok = True
            for property_name, func in self.filter_cons.items():
                # patch work to support vqpy.utils.continuing since Vobj needs
                # to be passed as an argument
                if type(func) == continuing:
                    ok = func(obj, property_name)
                else:
                    it = obj.getv(property_name)
                    if it is None or not func(it):
                        ok = False
                        break
            if ok:
                ret.append(obj)
        return ret

    def select(self, objs: List[VObjBaseInterface]) -> List[Dict]:
        """select the required informations from the constraints"""
        return [{key: postproc(x.getv(key))
                 for key, postproc in self.select_cons.items()}
                for x in objs]

    def apply(self, vobjs: List[VObjBaseInterface]) -> List[Dict]:
        """apply the constraint on a list of VObj instances"""
        # TODO: optimize the procedure
        filtered = self.filter(vobjs)
        selected = self.select(filtered)
        filtered_ids = [obj.getv("track_id") for obj in filtered]
        return selected, filtered_ids
