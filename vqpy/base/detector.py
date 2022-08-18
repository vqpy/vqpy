"""The detector base class"""

from typing import Dict, List

import numpy as np


class DetectorBase(object):
    """The base class of all predictors"""
    cls_names = None        # the class names of the classification
    output_fields = []      # the list of data fields the predictor can provide

    def inference(self, img: np.ndarray) -> List[Dict]:
        """Get the detected objects from the image
        img (np.ndarray): the inferenced images
        returns: list of objects, expressed in dictionaries
        """
        raise NotImplementedError
