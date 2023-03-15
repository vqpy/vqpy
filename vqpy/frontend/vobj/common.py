class UnComputedProperty(object):
    pass


def get_dep_properties(prop):
    vobj_properties = []
    if prop.is_vobj_property():
        all_dep_names = prop.inputs.keys()
        built_in_names = prop.vobj.get_builtin_property_names()
        vobj_dep_names = all_dep_names - built_in_names
        for name in vobj_dep_names:
            p = prop.vobj.get_property(name)
            vobj_properties.extend(get_dep_properties(p))
        vobj_properties.append(prop)

    return vobj_properties
