from collections import defaultdict


class Planner:
    def __init__(self):
        self.vobj_properties = dict()
        self.property_hist_len = defaultdict(int)

    def register_deps(self, vobj_types, deps):
        vobj_names = [vobj_type.__name__ for vobj_type in vobj_types]
        vobj_types = dict(zip(vobj_names, vobj_types))
        for attr, (dep, stateful) in deps.items():
            vobj_name, _ = attr.split(".")
            if vobj_name in vobj_types:
                self.vobj_properties[attr] = (dep, stateful)
                if stateful:
                    for dependent_attr, len in dep.items():
                        self.property_hist_len[dependent_attr] = max(
                            self.property_hist_len[dependent_attr], len
                        )
        # write hist_len to VObj, will be used by VObj property wrappers
        for attr, hist_len in self.property_hist_len.items():
            vobj_name, attr_name = attr.split(".")
            vobj_types[vobj_name].hist_len[attr_name] = hist_len
