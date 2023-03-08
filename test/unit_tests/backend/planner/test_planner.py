import vqpy

from typing import List
from icecream import ic


class VObj1(vqpy.VObjBase):
    @vqpy.stateless(inputs=["tlbr"])
    def attr1(self):
        pass

    @vqpy.stateful(inputs={"attr1": 30})
    def attr2(self):
        pass

    @vqpy.stateless(inputs=["attr2"])
    def attr3(self):
        pass


class VObj2(vqpy.VObjBase):
    @vqpy.stateless(inputs=["image"])
    def attr1(self):
        pass

    @vqpy.stateless(inputs=["VObj1.attr2"])
    def attr2(self):
        pass

    @vqpy.stateful(inputs={"attr1": 1, "attr2": 10, "attr3": (1, 1)})
    def attr3(self):
        pass


from vqpy.backend.planner.planner import Planner

cls_type = {"object1": VObj1, "object2": VObj2}


class Query(vqpy.QueryBase):
    """The class obtaining all fallen person"""

    @staticmethod
    def setting() -> List[vqpy.VObjConstraint]:
        filter_cons_VObj1 = {
            "__class__": lambda x: x == VObj1,
            "attr1": lambda x: x == "attr1_value",
            "attr3": lambda x: x == "attr3_value",
        }
        filter_cons_VObj2 = {
            "__class__": lambda x: x == VObj2,
            "attr3": lambda x: x == "attr3_value",
        }
        select_cons_VObj1 = {
            "attr2": None,
        }
        select_cons_VObj2 = {
            "attr1": None,
            "attr2": None,
        }
        vobj_constraint_1 = vqpy.VObjConstraint(
            filter_cons=filter_cons_VObj1,
            select_cons=select_cons_VObj1,
            filename="VObj1",
        )
        vobj_constraint_2 = vqpy.VObjConstraint(
            filter_cons=filter_cons_VObj2,
            select_cons=select_cons_VObj2,
            filename="VObj2",
        )
        # patchwork to specify type of VObj to filter in VObjConstraint
        setattr(vobj_constraint_1, "class", "VObj1")
        setattr(vobj_constraint_2, "class", "VObj2")
        return [vobj_constraint_1, vobj_constraint_2]


planner = Planner(cls_type)
ic(planner.plan(Query))
