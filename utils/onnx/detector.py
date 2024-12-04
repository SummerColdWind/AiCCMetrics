import numpy as np
from .transforms import Normalize, Compose
from onnxruntime import InferenceSession

from utils.common import CANVAS_SHAPE


class BaseDetector:

    def __init__(self, onnx_path):
        self.sess = InferenceSession(onnx_path)
        self.transform = Compose([Normalize()])

    def infer(self, image):
        inputs = self.transform({'img': image})['img']
        inputs = inputs[np.newaxis, ...]
        inputs = inputs[..., :CANVAS_SHAPE[0], :CANVAS_SHAPE[1]]
        ort_outs = self.sess.run(output_names=None, input_feed={self.sess.get_inputs()[0].name: inputs})
        return ort_outs

    def postprocess(self, ort_outs):
        pass

    def __call__(self, image):
        ort_outs = self.infer(image)
        image = self.postprocess(ort_outs)
        return image


class NerveSegmenter(BaseDetector):

    def __init__(self, onnx_path):
        super().__init__(onnx_path)

    def postprocess(self, ort_outs):
        image = np.squeeze(ort_outs[0]) * 255
        image = image.astype('uint8')
        return image
