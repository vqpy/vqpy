# flake8: noqa
import os
import cv2
import torch
import numpy as np

from SPPE.src.main_fast_inference import InferenNet_fast, InferenNet_fastRes50
from SPPE.src.utils.img import crop_dets
from pPose_nms import pose_nms
from SPPE.src.utils.eval import getPrediction


class SPPE_FastPose(object):
    def __init__(self,
                 backbone,
                 device='cuda',
                 weights_file=None):
        assert backbone in ['resnet50', 'resnet101'], '{} backbone is not support yet!'.format(backbone)

        self.device = device

        if backbone == 'resnet101':
            self.model = InferenNet_fast(weights_file=weights_file).to(device)
        else:
            self.model = InferenNet_fastRes50(weights_file=weights_file).to(device)
        self.model.eval()

    def predict(self, image, bboxs, inp_w, inp_h):
        inps, pt1, pt2 = crop_dets(image, bboxs, inp_h, inp_w)
        pose_hm = self.model(inps.to(self.device)).cpu().data

        # Cut eyes and ears.
        pose_hm = torch.cat([pose_hm[:, :1, ...], pose_hm[:, 5:, ...]], dim=1)

        xy_hm, xy_img, scores = getPrediction(pose_hm, pt1, pt2, inp_h, inp_w,
                                              pose_hm.shape[-2], pose_hm.shape[-1])
        results = pose_nms(bboxs, xy_img, scores)
        ps = results[0]
        keypoints = np.concatenate((ps['keypoints'].numpy(),ps['kp_score'].numpy()),axis=1)
        return keypoints