from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, List, Dict, Union
from queue import Queue

from vqpy.obj.vobj.deps import Dependency
from vqpy.query.vobj_constraint import VObjConstraint

from vqpy.backend.operator.vobj_filter import VObjFilter, VObjPropertyFilter
from vqpy.backend.operator.vobj_projector import VObjProjector
from vqpy.backend.operator.tracker import Tracker

# default properties from output of detector and tracker
default_properties = ["__class__", "tlbr", "image", "track_id"]
# initial properties from detector, bare minimum to create VObj
# need to run object tracker to obtain track_id, thus not considered "initial"
initial_properties = ["__class__", "tlbr", "image"]


@dataclass
class filter_predicate:
    filter_index: int = -1
    attr_name: str = None
    deps: Union[List, Dict] = None
    stateful: bool = False
    predicate: Callable = None


class Planner:
    def __init__(self, cls_type):
        vobj_types = list(set(cls_type.values()))
        vobj_names = [vobj_type.__name__ for vobj_type in vobj_types]
        vobj_types = dict(zip(vobj_names, vobj_types))
        # mapping from vobj type name to vobj type
        self.vobj_types = vobj_types
        self.vobj_names = vobj_names
        # mapping from vobj type name to detector's object name (e.g. COCO classes)
        self.vobj_names_mapping = defaultdict(list)
        for cls_name, vobj_type in cls_type.items():
            self.vobj_names_mapping[vobj_type.__name__].append(cls_name)
        # dependency of each property
        self.property_deps: Dict[str, Union[List, Dict]] = dict()
        # whether property is stateful
        self.property_stateful: Dict[str, bool] = dict()
        # rank of each property, after topo sort
        self.property_rank: Dict[str, int] = dict()

        self.parse_property_deps()

    def parse_property_deps(self):
        Dependency.update_hist_lens(vobj_types=self.vobj_types.values())
        # takes VObjBase, registered_names and property function dependencies required
        # return a dict of property -> rank
        for attr in Dependency.deps_in_use.keys():
            self.property_deps[attr] = Dependency.deps_in_use[attr]
            self.property_stateful[attr] = Dependency.stateful[attr]
        # manually add outputs from detector and tracker
        for vobj_type in self.vobj_types.keys():
            for attr_name in default_properties:
                self.property_deps[f"{vobj_type}.{attr_name}"] = dict()
                self.property_stateful[f"{vobj_type}.{attr_name}"] = False

        # build graph
        graph = defaultdict(list)
        in_degree = dict()
        for property, deps in self.property_deps.items():
            for dep in deps.keys():
                # TODO: workaround don't add dependency on itself
                if property != dep:
                    graph[dep].append(property)
            in_degree[property] = len(deps)
            # remove count for self dependency
            if property in deps.keys():
                in_degree[property] -= 1

        # sort
        queue = deque()
        for property, deps in self.property_deps.items():
            if len(deps) == 0:
                queue.append(property)

        sequence = []
        while queue:
            node = queue.popleft()
            sequence.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sequence) != len(self.property_deps):
            raise ValueError("Cycle detected in property dependencies")

        # apply rank values to sort results
        for rank, property in enumerate(sequence):
            self.property_rank[property] = rank

    def plan(self, query_task):
        vobj_constraints: List[VObjConstraint] = query_task.setting()
        filters: List[filter_predicate] = []
        # need to assign filter index to each vobj constraint
        for index, vobj_constraint in enumerate(vobj_constraints):
            filter_cons = vobj_constraint.filter_cons
            # TODO: workaround to get name of vobj type that vobj_constraint
            # is filtering
            vobj_name = getattr(vobj_constraint, "class")
            for attr_name, predicate in filter_cons.items():
                filters.append(
                    filter_predicate(
                        index,
                        f"{vobj_name}.{attr_name}",
                        self.property_deps[f"{vobj_name}.{attr_name}"],
                        self.property_stateful[f"{vobj_name}.{attr_name}"],
                        predicate,
                    )
                )
        # sort filters by property rank
        filters.sort(key=lambda x: self.property_rank[x.attr_name])

        operators = list()
        last_required_projector = 0
        computed = set()
        # add initial properties (outputs of detector)
        for vobj_name in self.vobj_names:
            computed.update(
                [f"{vobj_name}.{attr}" for attr in initial_properties]
            )
        # add necessary projectors before filters
        for filter in filters:
            to_adds = Queue()
            adds = list()  # pre-requisites of filter
            to_adds.put(filter.attr_name)
            while not to_adds.empty():
                to_add = to_adds.get()
                if not to_add in computed:
                    adds.append(to_add)
                    for dep in self.property_deps[to_add]:
                        # TODO: workaround, don't add dependency on itself
                        if dep != filter.attr_name:
                            to_adds.put(dep)
            adds.sort(key=lambda x: self.property_rank[x])
            computed.update(adds)

            # add projectors
            for add in adds:
                # track_id is obtained from tracker
                if add.split(".")[1] == "track_id":
                    operators.append(
                        Tracker(
                            prev=None,
                            class_name=add.split(".")[0],
                            filter_index=filter.filter_index,
                            fps=30,  # arbitrary value, required by ByteTracker
                        )
                    )
                else:
                    operators.append(
                        VObjProjector(
                            prev=None,
                            property_name=add.split(".")[1],
                            property_func=getattr(
                                self.vobj_types[add.split(".")[0]],
                                add.split(".")[1],
                            ),
                            dependencies=self.property_deps[add],
                            class_name=add.split(".")[0],
                            filter_index=filter.filter_index,
                        )
                    )
                    if Dependency.req_hist_len[add] > 1:
                        last_required_projector = len(operators) - 1
            # add filter
            filter_operator = None
            if filter.attr_name.split(".")[1] == "__class__":
                filter_operator = VObjFilter(
                    prev=None,
                    condition_func=self.vobj_names_mapping[
                        filter.attr_name.split(".")[0]
                    ],
                    filter_index=filter.filter_index,
                )
            else:
                filter_operator = VObjPropertyFilter(
                    prev=None,
                    property_name=filter.attr_name.split(".")[1],
                    property_condition_func=filter.predicate,
                    filter_index=filter.filter_index,
                )
            operators.append(filter_operator)

        # delay filters until all projectors required by stateful props are executed
        filter_operators = []
        filter_operator_indices = []
        for i in range(last_required_projector):
            if type(operators[i]) == VObjPropertyFilter:
                filter_operator_indices.append(i)
                filter_operators.append(operators[i])

        for i in range(len(filter_operator_indices)):
            operators.insert(last_required_projector + 1, filter_operators[i])
            operators.pop(filter_operator_indices[i] - i)

        return operators
