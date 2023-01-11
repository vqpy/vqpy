from __future__ import annotations

from typing import Callable, Dict, List, Optional
from ..base.interface import \
    VObjBaseInterface, VObjConstraintInterface, FrameInterface
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

    def filter(self, frame: FrameInterface) -> List[VObjBaseInterface]:
        """filter the list of vobjects from the constraint"""
        # Some assumptions:
        # - each query only filters one type of VObj (the desired VObj type)
        # - queries will implement such filtering of a single type of VObj
        #   by adding restriction on the "__class__" property

        # function used in query to select the desired VObjs
        filter_func = self.filter_cons["__class__"]
        # get the first VObj type that satisfies the filter function,
        # should be the desired VObj type
        vobj_type = next((
                vobj_type
                for vobj_type in frame.vobjs.keys() if filter_func(vobj_type)
            ),
            None
        )
        # all VObjs of the desired type
        vobjs = frame.get_tracked_vobjs(vobj_type)

        # compute cross_vobj_property's in the VObjs of the desired type
        self._compute_cross_vobj_property(
            frame, vobjs, self.filter_cons.keys()
        )

        ret: List[VObjBaseInterface] = []
        for obj in vobjs:
            ok = True
            for property_name, func in self.filter_cons.items():
                if property_name == "__class__":
                    # skip filters about "__class__" since we already used it
                    # to only include VObjs of the desired type
                    continue
                if type(func) == continuing:
                    # patch work to support vqpy.utils.continuing since
                    # VObj and the property name need to be passed as arguments
                    ok = func(obj, property_name)
                else:
                    it = obj.getv(property_name)
                    if it is None or not func(it):
                        ok = False
                        break
            if ok:
                ret.append(obj)
        return ret

    def select(self, objs: List[VObjBaseInterface], frame: FrameInterface) \
            -> List[Dict]:
        """select the required informations from the constraints"""
        # compute cross_vobj_property's in the filtered VObjs, for potential
        # usage in select_cons
        # select_cons are provided so only those being used will be computed
        self._compute_cross_vobj_property(
            frame, objs, self.select_cons.keys() - self.filter_cons.keys()
        )
        return [{key: postproc(x.getv(key))
                 for key, postproc in self.select_cons.items()}
                for x in objs]

    def apply(self, frame: FrameInterface) -> List[Dict]:
        """apply the constraint on a list of VObj instances"""
        # TODO: optimize the procedure
        filtered = self.filter(frame)
        selected = self.select(filtered, frame)
        filtered_ids = [obj.getv("track_id") for obj in filtered]
        return selected, filtered_ids

    def _compute_cross_vobj_property(
        self, frame: FrameInterface, vobjs: VObjBaseInterface, property_names
    ) -> None:
        """Compute cross_vobj_property's used in the conditions for the given
        VObjs"""
        if len(vobjs) == 0:
            return

        # Retrieve properties of other VObjs that are
        # 1. required by @cross_vobj_property 2. used in filter/select_cons
        # and store them in a dictionary
        # Dict{
        #     cross_vobj_property_name: \
        #         List[
        #             Tuple(required vobj1's property1, property2, ...),
        #             Tuple(required vobj2's properties),
        #             ...
        #         ],
        #     ...
        # }
        # Since all VObjs of the same type should share the same properties
        # within each query, we only need to retrieve the values of the
        # properties once, using _registered_cross_vobj_names from any VObj
        # of that type (use the first one here)
        cross_vobj_args = {}
        for cross_vobj_property in \
                vobjs[0]._registered_cross_vobj_names.keys():
            # only compute properties used in the conditions
            if cross_vobj_property not in property_names:
                continue
            other_vobj_type, other_vobj_input_fields = \
                vobjs[0]._registered_cross_vobj_names[cross_vobj_property]
            other_vobjs = frame.get_tracked_vobjs(other_vobj_type)
            properties = []
            for other_vobj in other_vobjs:
                properties.append(
                    tuple(
                        other_vobj.getv(input_field)
                        for input_field in other_vobj_input_fields
                    )
                )
            cross_vobj_args[cross_vobj_property] = properties

        # for each vobj, compute value of cross_vobj_property
        for obj in vobjs:
            for property_name in property_names:
                if property_name in obj._registered_cross_vobj_names.keys():
                    getattr(obj, property_name)(cross_vobj_args[property_name])
