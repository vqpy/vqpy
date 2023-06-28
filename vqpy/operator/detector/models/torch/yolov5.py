"""
The yolov5 detector for object detection, from ultralytics
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from loguru import logger
from vqpy.operator.detector.base import DetectorBase
from vqpy.class_names.coco import COCO_CLASSES


class YoloV5Detector(DetectorBase):
    """The YOLOV5 detector for object detection"""

    cls_names = COCO_CLASSES
    output_fields = ["class_id", "tlbr", "score"]

    def __init__(self, **kwargs):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5x").cuda()

    def inference(self, img) -> List[Dict]:
        # frames = torch.tensor([img])
        # frames = torch.permute(frames, (0, 2, 3, 1))
        # predictions = self.model(
        #     [its.cpu().detach().numpy() for its in frames]
        # )
        # prediction = predictions.pandas().xyxy[0]
        # pred_class = prediction["name"].tolist()
        # pred_score = prediction["confidence"].tolist()
        # pred_boxes = prediction[["xmin", "ymin", "xmax", "ymax"]].apply(
        #     lambda x: list(x), axis=1
        # )
        results = self.model(img)

        df = pd.DataFrame(results.pandas().xyxy[0])
        tlbr = df[["xmin", "ymin", "xmax", "ymax"]].values
        score = df["confidence"].values
        class_id = df["class"].values
        rets = [
            {
                "tlbr": np.asarray(tlbr[i]),
                "score": score[i],
                "class_id": int(class_id[i]),
            }
            for i in range(len(tlbr))
        ]
        if len(rets) == 0:
            return []
        return rets
