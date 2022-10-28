from vqpy.detector.utils import onnx_inference
from vqpy.base.detector import DetectorBase
from vqpy.utils.classes import COCO_CLASSES
import numpy as np
from typing import Dict, List
import cv2
from vqpy.detector.logger import register


class FasterRCNNDdetector(DetectorBase):
    """The YOLOX detector for object detection"""

    cls_names = COCO_CLASSES
    output_fields = ["tlbr", "score", "class_id"]

    def inference(self, img: np.ndarray) -> List[Dict]:
        processed_img = preprocess(img)
        detections = onnx_inference(processed_img, self.model_path)
        outputs = postprocess(detections, img.shape)
        return outputs


def preprocess(image):
    # Resize
    ratio = 800.0 / min(image.shape[0], image.shape[1])
    image = cv2.resize(
        image,
        (int(ratio * image.shape[1]), int(ratio * image.shape[0])),
        interpolation=cv2.INTER_LINEAR).astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image


def postprocess(detections, image_size):
    boxes, labels, scores = detections
    # Resize boxes
    ratio = 800.0 / min(image_size[0], image_size[1])
    boxes /= ratio

    rets = []
    for (tlbr, score, class_id) in zip(boxes, scores, labels):
        # todo: convert dict to named tuple
        rets.append({"tlbr": np.asarray(tlbr),
                     "score": score.item(),
                     "class_id": class_id - 1})
    return rets


register("faster_rcnn", FasterRCNNDdetector, "FasterRCNN-10.onnx")
