def get_dep_properties(prop):
    vobj_properties = []
    vobj_properties_names = set()
    if prop.is_vobj_property():
        all_dep_names = prop.inputs.keys()
        built_in_names = prop.vobj.get_builtin_property_names()
        vobj_dep_names = all_dep_names - built_in_names
        for name in vobj_dep_names:
            if name == prop.name:
                continue
            p = prop.vobj.get_property(name)
            dep_properties = get_dep_properties(p)
            # remove duplicates
            for dep_prop in dep_properties:
                if dep_prop.name not in vobj_properties_names:
                    vobj_properties.append(dep_prop)
                    vobj_properties_names.add(dep_prop.name)
        vobj_properties.append(prop)
        vobj_properties_names.add(prop.name)

    return vobj_properties
