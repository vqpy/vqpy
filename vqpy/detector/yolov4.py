from vqpy.detector.utils import onnx_inference
from vqpy.base.detector import DetectorBase
from vqpy.utils.classes import COCO_CLASSES
import numpy as np
from typing import Dict, List
import cv2
from scipy import special
from vqpy.detector.logger import register


MODEL_INPUT_SIZE = (416, 416)
STRIDES = [8, 16, 32]
XYSCALE = [1.2, 1.1, 1.05]
ANCHORS = np.array([12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110,
                    192, 243, 459, 401], dtype=np.float32).reshape(3, 3, 2)


class Yolov4Detector(DetectorBase):
    """The YOLOX detector for object detection"""

    cls_names = COCO_CLASSES
    output_fields = ["tlbr", "score", "class_id"]

    def inference(self, img: np.ndarray) -> List[Dict]:
        processed_img = preprocess(img)
        detections = onnx_inference(processed_img, self.model_path)
        outputs = postprocess(detections, img.shape)
        return outputs


def preprocess(image):
    # this function is from tensorflow-yolov4-tflite/core/utils.py
    image = np.copy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ih, iw = MODEL_INPUT_SIZE
    h, w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_padded = image_padded / 255.
    image_data = image_padded[np.newaxis, ...].astype(np.float32)
    return image_data


def postprocess(detections, image_size):
    def postprocess_bbbox(pred_bbox):
        '''define anchor boxes'''
        for i, pred in enumerate(pred_bbox):
            conv_shape = pred.shape
            output_size = conv_shape[1]
            conv_raw_dxdy = pred[:, :, :, :, 0:2]
            conv_raw_dwdh = pred[:, :, :, :, 2:4]
            xy_grid = np.meshgrid(np.arange(output_size),
                                  np.arange(output_size))
            xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)

            xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
            xy_grid = xy_grid.astype(np.float)

            pred_xy = ((special.expit(conv_raw_dxdy) * XYSCALE[i]) -
                       0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
            pred_wh = (np.exp(conv_raw_dwdh) * ANCHORS[i])
            pred[:, :, :, :, 0:4] = np.concatenate([pred_xy, pred_wh], axis=-1)

        pred_bbox = [np.reshape(x, (-1, np.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = np.concatenate(pred_bbox, axis=0)
        return pred_bbox

    def postprocess_boxes(pred_bbox, org_img_shape, score_threshold):
        '''remove boundary boxs with a low detection probability'''
        valid_scale = [0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5],
                                   axis=-1)
        # # (2) (xmin, ymin, xmax, ymax) ->
        # # (xmin_org, ymin_org, xmax_org, ymax_org)
        org_h, org_w = org_img_shape[0], org_img_shape[1]
        ih, iw = MODEL_INPUT_SIZE
        resize_ratio = min(iw / org_w, ih / org_h)

        dw = (iw - resize_ratio * org_w) / 2
        dh = (ih - resize_ratio * org_h) / 2

        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # # (3) clip some boxes that are out of range
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:],
                                               [org_w - 1, org_h - 1])],
                                   axis=-1)
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]),
                                     (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # # (4) discard some invalid boxes
        value = np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2],
                                   axis=-1)
        bboxes_scale = np.sqrt(value)
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale),
                                    (bboxes_scale < valid_scale[1]))

        # # (5) discard some boxes with low scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = np.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        return np.concatenate([coors, scores[:, np.newaxis],
                               classes[:, np.newaxis]], axis=-1)

    def bboxes_iou(boxes1, boxes2):
        '''calculate the Intersection Over Union value'''
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_w = (boxes1[..., 2] - boxes1[..., 0])
        boxes1_h = (boxes1[..., 3] - boxes1[..., 1])
        boxes1_area = boxes1_w * boxes1_h
        boxes2_w = (boxes2[..., 2] - boxes2[..., 0])
        boxes2_h = (boxes2[..., 3] - boxes2[..., 1])
        boxes2_area = boxes2_w * boxes2_h

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = np.maximum(1.0 * inter_area / union_area,
                          np.finfo(np.float32).eps)

        return ious

    def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
        """
        :param bboxes: (xmin, ymin, xmax, ymax, score, class)

        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
            https://github.com/bharatsingh430/soft-nms
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]

            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind],
                                             cls_bboxes[max_ind + 1:]])
                iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes

    pred_bbox = postprocess_bbbox(detections)
    bboxes = postprocess_boxes(pred_bbox, image_size, score_threshold=0.25)
    bboxes = nms(bboxes, 0.213, method='nms')

    rets = []
    for bbox in bboxes:
        rets.append({"tlbr": np.asarray(bbox[:4]),
                     "score": bbox[4],
                     "class_id": int(bbox[5])})
    return rets


register("yolov4", Yolov4Detector, "yolov4.onnx")
