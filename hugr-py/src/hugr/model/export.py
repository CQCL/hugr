import hugr.model as model
from hugr.hugr.base import Hugr, Node


class Export:
    def __init__(self, hugr: Hugr):
        self.hugr = hugr

    def export_node(self, node: Node):
        node_data = self.hugr[node]
        pass
