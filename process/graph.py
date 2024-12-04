import itertools
from functools import lru_cache

from utils.common import distance, get_canvas, close, calculate_angle, split
from process.skeleton import get_skeleton
from process.instance import NerveContainer
from process.point import get_points

import cv2
import numpy as np
import networkx as nx


class Trunk:
    def __init__(self, segments, nodes, graph, start_node, end_node):
        self.segments = segments
        self.nodes = nodes
        self.graph = graph
        self.start_node = start_node
        self.end_node = end_node

        self._body = None
        self._bone = None
        self._corner = None

    def __repr__(self):
        return f"""Trunk:
        Segments: {self.segments}
        Nodes: {self.nodes}
        Start: {self.start_node}, End: {self.end_node}
        Length: {self.length}, Linearity: {self.linearity}
        """

    @property
    def body(self):
        if self._body is None:
            canvas = get_canvas()
            for segment in self.segments:
                canvas[segment.body > 0] = 255
            iteration = 1
            while True:
                canvas = close(canvas, iteration=iteration)
                _, num = split(canvas)
                if num == 1:
                    break
                iteration += 1
            self._body = canvas
        return self._body

    @property
    def bone(self):
        if self._bone is None:
            self._bone = get_skeleton(self.body)
        return self._bone

    @property
    def corner(self):
        if self._corner is None:
            start_point = np.transpose(np.nonzero(get_points(self.bone)[1]))[0]
            points = np.transpose(np.nonzero(self.bone))
            curve = np.array(sorted(points, key=lambda x: distance(x, start_point)))[:, ::-1]
            epsilon = 5.0
            approx_curve = cv2.approxPolyDP(curve, epsilon, True)
            angles = []
            for i in range(1, len(approx_curve) - 1):
                p1 = approx_curve[i - 1][0]
                p2 = approx_curve[i][0]
                p3 = approx_curve[i + 1][0]
                angle = calculate_angle(p1, p2, p3)
                angles.append(angle)
            self._corner = angles
        return self._corner

    @property
    def min_corner(self):
        return np.min(self.corner) if self.corner else 180

    @property
    def mean_corner(self):
        return np.mean(self.corner) if self.corner else 180


    @property
    def length(self):
        return np.sum([s.length for s in self.segments])

    @property
    def linearity(self):
        if self.length:
            return distance(self.nodes[self.start_node].center, self.nodes[self.end_node].center) / self.length
        return 0

    @property
    def area(self):
        return cv2.countNonZero(self.body)



def get_graph(segments, nodes):
    graph = nx.Graph()
    edges = [[n.index for n in s.neighbors] + [s.length, s.index] for s in segments if len(s.neighbors) == 2]
    for u, v, weight, edge_id in edges:
        graph.add_edge(u, v, weight=weight, edge_id=edge_id)

    return graph


def get_shortest(graph, start_node, end_node):
    shortest_nodes = nx.dijkstra_path(graph, start_node, end_node)
    shortest_length = nx.dijkstra_path_length(graph, start_node, end_node)
    shortest_edges = []
    for i in range(len(shortest_nodes) - 1):
        u, v = shortest_nodes[i], shortest_nodes[i + 1]
        edge_data = graph.get_edge_data(u, v)
        shortest_edges.append(edge_data['edge_id'])


    return shortest_edges, shortest_nodes, shortest_length



def filter_trunk(trunks, nodes):
    m = [x.index for x in nodes]
    subsets = [([n.index for n in t.nodes], t.min_corner * t.area) for t in trunks]
    subset_indices = {tuple(subset): x for subset, x in subsets}
    subset_masks = {i: 0 for i in range(len(subset_indices))}

    for i, (subset, _) in enumerate(subsets):
        subset_masks[i] = sum((1 << m.index(num)) for num in subset)

    @lru_cache(None)
    def dp(mask):
        max_avg = 0
        best_subset = -1
        count = 0
        for i in range(len(subset_masks)):
            if mask & subset_masks[i] == 0:
                current_avg, _, current_count = dp(mask | subset_masks[i])
                current_avg = (current_avg * current_count + subset_indices[tuple(subsets[i][0])]) / (current_count + 1)
                if current_avg > max_avg:
                    max_avg = current_avg
                    best_subset = i
                    count = current_count + 1
        return max_avg, best_subset, count

    max_avg, _, _ = dp(0)

    # Trace back to find the subsets used
    used_subsets = []
    mask = 0
    while True:
        _, best_subset, _ = dp(mask)
        if best_subset == -1:
            break
        used_subsets.append(subsets[best_subset][0])
        mask |= subset_masks[best_subset]

    return used_subsets

def get_trunk(segments, nodes):
    trunk_list = []
    graph = get_graph(segments, nodes)
    shortest_dict = {}
    for start_node, end_node in itertools.combinations([n.index for n in nodes if n.class_node == 'end'], 2):
        shortest_edges, shortest_nodes, length = get_shortest(graph, start_node, end_node)
        shortest_dict[tuple(shortest_nodes)] = shortest_edges

        trunk = Trunk(
            NerveContainer([segments[i] for i in shortest_edges]),
            NerveContainer([nodes[i] for i in shortest_nodes]),
            graph,
            start_node,
            end_node
        )

        if trunk.linearity > .8 and trunk.length > 200 and trunk.min_corner > 75:
            trunk_list.append(trunk)

    trunks = filter_trunk(trunk_list, nodes)
    return [(shortest_dict[tuple(t)], t) for t in trunks] if trunks else None
