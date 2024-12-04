from utils.common import show_image, load_image, save_image, get_canvas, split, dilated
from process.skeleton import get_skeleton

import cv2
import numpy as np


class NerveSegment:
    def __init__(self, index, body, bone, class_='Segment'):
        self.index = index
        self.body = body
        self.bone = bone
        self.class_ = class_  # 'Segment' or 'Node'
        self.neighbors = []
        self.class_node = None  # 'end' or 'branch'
        self.class_segment = 'branch'  # 'main' or 'branch'

        # property
        self._area = None
        self._area_weighted = None
        self._length = None
        self._width = None
        self._width_weighted = None
        self._ends = None
        self._center = None

    def __repr__(self):
        return f'{self.class_} {self.index}'

    @property
    def center(self):
        if self._center is None:
            indices = np.argwhere(self.bone != 0)
            center = np.mean(indices, axis=0)
            self._center = int(center[1]), int(center[0])
        return self._center  # As (x, y)

    @property
    def length(self):
        if self._length is None:
            length = 0
            contours, _ = cv2.findContours(self.bone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                length += cv2.arcLength(c, True) / 2
            self._length = length
        return self._length

    @property
    def area(self):
        if self._area is None:
            self._area = cv2.countNonZero(self.body)
        return self._area

    @property
    def width(self):
        if self._width is None:
            self._width = self.area / self.length
        return self._width


class NerveContainer:
    def __init__(self, instance):
        self.instance = instance


    def __repr__(self):
        return str([i.index for i in self.instance])

    def __getitem__(self, item):
        try:
            return [x for x in self.instance if x.index == item][0]
        except IndexError:
            raise IndexError('Not Found.')

    def __iter__(self):
        return iter(self.instance)


def _deconvolution(skeleton, image, threshold=3):
    skeleton = skeleton > 0
    skeleton = skeleton.astype('bool')

    dist_transform = cv2.distanceTransform((1 - skeleton).astype(np.uint8), cv2.DIST_L2, 5)
    distance_threshold = threshold
    selected_pixels = (dist_transform <= distance_threshold) & image

    segments, num = split(selected_pixels, False)
    if num == 1:
        return selected_pixels
    else:
        areas = [cv2.countNonZero(s) for s in segments]
        return segments[np.argmax(areas)]


def _edge(image):
    image2 = dilated(image, np.ones((3, 3), dtype='uint8'))
    edge = np.bitwise_xor(image, image2)
    return edge



def get_instance(image, skeleton, points, end_points):
    segments_list, nodes_list = [], []

    canvas = get_canvas()
    canvas[skeleton > 0] = 255
    canvas[points > 0] = 0
    segments, _ = split(canvas, split_skeleton=True)
    for i, s in enumerate(segments):
        segment = _deconvolution(s, image)
        segments_list.append(NerveSegment(
            i,
            segment.copy(),
            s.copy(),
            'Segment')
        )

    points = cv2.bitwise_or(points, end_points)
    nodes, _ = split(points, split_skeleton=True)
    for i, n in enumerate(nodes):
        node = NerveSegment(
            i,
            None,
            n,
            'Node'
        )

        node.class_node = 'end' if cv2.countNonZero(cv2.bitwise_and(node.bone, end_points)) else 'branch'
        nodes_list.append(node)
        edge = _edge(n)
        for segment in segments_list:
            intersection = cv2.bitwise_and(edge, segment.bone)
            if cv2.countNonZero(intersection) > 0:
                node.neighbors.append(segment)
                if node not in segment.neighbors:
                    segment.neighbors.append(node)


    return NerveContainer(segments_list), NerveContainer(nodes_list)
