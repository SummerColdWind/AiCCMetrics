from skimage.morphology import skeletonize
from utils.common import split

import cv2

def get_skeleton(image):
    image = image > 0
    skeleton = skeletonize(image)
    skeleton = skeleton.astype('uint8')
    skeleton = skeleton * 255

    # 设置边缘像素为0
    skeleton[0, :] = 0  # 上边缘
    skeleton[-1, :] = 0  # 下边缘
    skeleton[:, 0] = 0  # 左边缘
    skeleton[:, -1] = 0  # 右边缘

    # 过滤噪点
    segments, _ = split(skeleton, True)

    for segment in segments:
        if cv2.countNonZero(segment) < 8:
            skeleton[segment > 0] = 0

    return skeleton


