from typing import Dict, List

import numpy as np


class DetectorBase(object):
    """The base class of all predictors"""
    cls_names = None        # the class names of the classification
    output_fields = []      # the list of data fields the predictor can provide
    
    def inference(self, img: np.ndarray) -> List[Dict]:
        raise NotImplementedError
