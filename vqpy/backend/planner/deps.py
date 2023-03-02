# dict used to temporarily store dependencies between VObj properties
# until planner takes them and generate dependency graph of VObj properties

# Ideally this Dependency dict and dependency graph in planner should be reset
# once planner finishes using it. However, VObj classes could be reused between
# vqpy.launch queries, so we update Dependency dict when VObj is re-declared,
# and only use dependencies between properties of VObjs mentioned in
# vqpy.launch

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

Dependency = dict()
