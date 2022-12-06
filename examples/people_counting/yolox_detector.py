"""
Based on demo implementation in Megvii YOLOX repo
The YOLOX detector for object detection
"""

from typing import Dict, List

import numpy as np
import torch
from loguru import logger
from vqpy.base.detector import DetectorBase
from vqpy.utils.classes import COCO_CLASSES

from yolox.data.data_augment import ValTransform
from yolox.exp.build import get_exp
from yolox.utils import postprocess
from yolox.utils.model_utils import get_model_info


class YOLOXDetector(DetectorBase):
    """The YOLOX detector for object detection"""

    cls_names = COCO_CLASSES
    output_fields = ["tlbr", "score", "class_id"]

    def __init__(self, model_path, device="gpu", fp16=True):
        # TODO: start a new process handling this
        exp = get_exp(None, "yolox_x")
        exp.test_conf = 0.3
        exp.nmsthre = 0.3
        exp.test_size = (640, 640)

        model = exp.get_model()
        model_info = get_model_info(model, exp.test_size)
        logger.info(f"Model Summary: {model_info}")
        if device == 'gpu':
            model.cuda()
            if fp16:
                model.half()
        model.eval()

        logger.info("loading checkpoint")
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=False)
        self.postproc = postprocess

    def inference(self, img) -> List[Dict]:
        ratio = min(self.test_size[0] / img.shape[0],
                    self.test_size[1] / img.shape[1])

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.model(img)
            outputs = self.postproc(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )

        outputs = outputs[0]
        if outputs is None:
            return []
        bboxes = (outputs[:, 0:4] / ratio).cpu()
        scores = (outputs[:, 4:5] * outputs[:, 5:6]).cpu()
        cls = outputs[:, 6:7].cpu()

        rets = []
        for (tlbr, score, class_id) in zip(bboxes, scores, cls):
            rets.append({"tlbr": np.asarray(tlbr),
                         "score": score.item(),
                         "class_id": int(class_id.item())})
        return rets
