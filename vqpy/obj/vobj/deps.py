from typing import List
from collections import defaultdict

from vqpy.obj.vobj.base import VObjBase


class Dependency:
    # Dict used to temporarily store dependencies between VObj properties
    # until we get list of VObj that will be used in the query from vqpy.launch
    # format: dictionary
    # from
    # f"{vobj_name}.{attr_name}"
    # to
    # Tuple(list/dict of deps, boolean of stateful/stateless)
    # example:
    # {
    #     "Person.keypoints": (["Person.image"], False),
    #     "Person.pose": ({"Person.keypoints": 30}, True),
    # }
    registered_deps = dict()
    # Dependency between property of VObjs that are used in the query.
    # VObj classes could be reused between vqpy.launch queries, so we update
    # Dependency dict when VObj is declared, and only use dependencies between
    # properties of VObjs mentioned in vqpy.launch
    # format: same as registered_deps
    deps_in_use = dict()
    # dict storing minimal length of VObj property value to store
    # format: f"{vobj_name}.{attr_name}" -> int
    property_hist_len = defaultdict(int)

    @classmethod
    # Register dependency during VObj type declaration
    def register_dep(cls, attr, deps):
        cls.registered_deps[attr] = deps

    @classmethod
    # update hist lens requirement of VObj property with list of VObj types
    # used in query. Invoke at beginning of vqpy.launch
    def update_hist_lens(cls, vobj_types: List[VObjBase]):
        cls.deps_in_use = dict()
        cls.property_hist_len = defaultdict(int)
        vobj_names = [vobj_type.__name__ for vobj_type in vobj_types]
        vobj_types = dict(zip(vobj_names, vobj_types))
        for attr, (dep, stateful) in cls.registered_deps.items():
            vobj_name, _ = attr.split(".")
            if vobj_name in vobj_types:
                cls.deps_in_use[attr] = (dep, stateful)
                if stateful:
                    for dependent_attr, len in dep.items():
                        cls.property_hist_len[dependent_attr] = max(
                            cls.property_hist_len[dependent_attr], len
                        )
        # write hist_len to VObj, will be used by VObj property wrappers
        for attr, hist_len in cls.property_hist_len.items():
            vobj_name, attr_name = attr.split(".")
            vobj_types[vobj_name].hist_len[attr_name] = hist_len
