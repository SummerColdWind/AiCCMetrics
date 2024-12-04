class Nerve:
    def __init__(self, segments, nodes):
        self.segments = segments
        self.nodes = nodes

    def __repr__(self):
        return f"""Nerve
        Segments: {[s.index for s in self.segments]}
        Nodes: {[n.index for n in self.nodes]}
        """


def get_nerve(segments, nodes):
    return Nerve(segments, nodes)




