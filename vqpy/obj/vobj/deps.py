from typing import List, Dict
from collections import defaultdict

from vqpy.obj.vobj.base import VObjBase


class Dependency:
    # dict of deps. format: {
    #     f"{vobj_name}.{attr_name}" ->
    #     {attr: (oldest frame_id, latest frame_id)}
    # }
    # range (oldest frame_id, latest frame_id) is (exclusive, inclusive],
    # e.g. (1,0) requires current frame and the frame before, (0,0) requires
    # current frame only (stateless)
    registered_deps: Dict[str, Dict[str, int]] = defaultdict(dict)
    # whether property is stateful. format: f"{vobj_name}.{attr_name}" -> bool
    stateful: Dict[str, bool] = dict()
    # also dict of deps, but only for properties of VObjs used in the query
    # VObj classes could be reused between vqpy.launch queries, so we update
    # Dependency dict when VObj is declared, and only use dependencies between
    # properties of VObjs mentioned in vqpy.launch
    # format same as registered_deps
    deps_in_use: Dict[str, Dict[str, int]] = dict()
    # dict storing minimal length of VObj property value to store
    # format: f"{vobj_name}.{attr_name}" -> int
    req_hist_len = defaultdict(int)

    @classmethod
    # Register dependency during VObj type declaration
    def register_dep(cls, vobj_name, attr_name, deps, stateful):
        for prereq_attr, hist_req in deps.items():
            # add vobj_name to attr if not specified (default to current VObj)
            if "." not in prereq_attr:
                prereq_attr = f"{vobj_name}.{prereq_attr}"
            # hist_range is (hist_req , 0] if user specifies hist_req as int
            hist_range = (hist_req, 0) if type(hist_req) is int else hist_req
            cls.registered_deps[f"{vobj_name}.{attr_name}"][
                prereq_attr
            ] = hist_range
        cls.stateful[f"{vobj_name}.{attr_name}"] = stateful
        # add track_id to deps if stateful
        # (1,0) is range (1,0], current frame only
        if stateful:
            cls.registered_deps[f"{vobj_name}.{attr_name}"][
                f"{vobj_name}.track_id"
            ] = (1, 0)

    @classmethod
    # update hist lens requirement of VObj property with list of VObj types
    # used in query. Invoke at beginning of vqpy.launch
    def update_hist_lens(cls, vobj_types: List[VObjBase]):
        # reset for new vqpy.launch
        cls.deps_in_use = defaultdict(dict)
        cls.req_hist_len = defaultdict(int)
        vobj_names = [vobj_type.__name__ for vobj_type in vobj_types]
        vobj_types = dict(zip(vobj_names, vobj_types))
        for attr, dep in cls.registered_deps.items():
            vobj_name, _ = attr.split(".")
            if vobj_name in vobj_names:
                for prereq_attr, hist_len in dep.items():
                    # need to store history since oldest frame_id
                    cls.deps_in_use[attr][prereq_attr] = (
                        cls.registered_deps[attr][prereq_attr][0] - 1
                    )  # VObjProjector excludes current frame from history len
                    # TODO: change VObjProjector to use range instead of len
                    cls.req_hist_len[prereq_attr] = max(
                        cls.req_hist_len[prereq_attr],
                        cls.registered_deps[attr][prereq_attr][0],
                    )
        # write hist_len to VObj, will be used by VObj property wrappers
        for attr, hist_len in cls.req_hist_len.items():
            vobj_name, attr_name = attr.split(".")
            vobj_types[vobj_name].hist_len[attr_name] = hist_len
