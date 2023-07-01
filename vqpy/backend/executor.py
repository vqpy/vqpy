from vqpy.backend.planner import add_video_metadata


class Executor:

    def __init__(self, root_plan_node, launch_args):
        self.root_plan_node = root_plan_node
        self.launch_args = add_video_metadata(launch_args)
        self.root_operator = root_plan_node.to_operator(self.launch_args)

    def execute(self):
        while self.root_operator.has_next():
            result = self.root_operator.next()
            yield result
