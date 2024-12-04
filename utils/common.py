import cv2
import numpy as np
from scipy.ndimage import label
import math
import re

# 常量
CANVAS_SHAPE = (384, 384)
DILATED_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
STRUCTURE_4 = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])

STRUCTURE_8 = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]])
CHINESE_PATTERN = re.compile(r'[\u4e00-\u9fa5]+')

def show_image(image):
    image_show = image.copy().astype('uint8')
    if np.amax(image_show) == 1:
        image_show = image_show * 255
    cv2.imshow('Show', image_show)
    cv2.waitKey(0)


def load_image(path):
    with open(path, 'rb') as file:
        file_data = file.read()
        image_array = np.frombuffer(file_data, np.uint8)
        image_raw = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image_raw


def save_image(image, path='result.png'):
    extend = '.' + path.split('.')[-1]
    retval, buffer = cv2.imencode(extend, image.astype('uint8'))
    with open(path, 'wb') as f:
        f.write(buffer)


def get_canvas(channels=1):
    if channels > 1:
        return np.zeros([*CANVAS_SHAPE, channels], dtype='uint8')
    return np.zeros(CANVAS_SHAPE, dtype='uint8')


def dilated(image, kernel=None, iteration=1):
    dilated_image = cv2.dilate(image, DILATED_KERNEL if kernel is None else kernel, iterations=iteration)
    return dilated_image


def close(image, kernel=None, iteration=1):
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel=CLOSE_KERNEL if kernel is None else kernel, iterations=iteration)
    return closed_image


def distance(p1, p2):
    p1, p2 = np.array(p1), np.array(p2)
    return np.linalg.norm(p1 - p2)


def split(image, split_skeleton=False):
    image = image > 0
    if not split_skeleton:
        arrays, num = label(image, structure=STRUCTURE_4)
    else:
        arrays, num = label(image, structure=STRUCTURE_8)

    segments = []
    for i in range(1, num + 1):
        component_image = np.where(arrays == i, 1, 0)
        component_image = component_image * 255
        component_image = component_image.astype('uint8')
        segments.append(component_image)

    return segments, num


def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    try:
        angle = math.degrees(math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    except ValueError:
        angle = 180
    return angle


def extract_Chinese(string, join=False):
    chinese_string = CHINESE_PATTERN.findall(string)
    if chinese_string:
        return ''.join(chinese_string) if join else chinese_string[0]
    return ''


