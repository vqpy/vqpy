from vqpy.backend.operator.base import Operator
from vqpy.backend.frame import Frame
from typing import Set, Union, Optional
from collections import defaultdict
from vqpy.operator.detector import vqpy_detectors
import os
import torch


class ObjectDetector(Operator):
    def __init__(self,
                 prev: Operator,
                 class_names: Union[str, Set[str]],
                 detector_name: Optional[str] = None,
                 detector_kwargs: dict = dict(),
                 ):
        """Object detector Operator.
        It uses the built-in object detector with name of {detector_name}
        for detecting interested classes defined in {class_names}. It also
        generates the `vobj_data` field in `frame`, which contains the
        detected vobjs of interested class, where each vobj is a dictionary
        of detection outputs (e.g. "tlbr", "score").

        Args:
            prev (Operator): The previous operator instance.
            class_names: One or multiple class names that users are interested.
                         Note that it should be a subset of the class names
                        supported by the detector with {detector_name}.
            detector_name: Oject detector name. e.g. "yolox".
                           Defaults to None.
            detector_kwargs (dict, optional): Additional arguments for object
                        detector . Defaults to dict().
        """
        self.prev = prev

        self._check_set_class_names(class_names)
        self.detector = self._setup_detector(detector_name, detector_kwargs)
        self.detector_name = detector_name

    def _check_set_class_names(self, class_names):
        if isinstance(class_names, str):
            self.class_names = {class_names}
        elif isinstance(class_names, set):
            self.class_names = class_names
        else:
            raise ValueError(f"Invalid class_names: {class_names}, which "
                             f"should be either a string or a set of strings.")

    def _setup_detector(self, detector_name, detector_kwargs):
        # check whether detector_name is valid
        if detector_name not in vqpy_detectors:
            raise ValueError(f"Detector name of {detector_name} hasn't been"
                             f"registered to VQPy")

        # create detector
        detector_type, weights_path, url = vqpy_detectors[detector_name]
        if not os.path.exists(weights_path):
            if url is not None:
                torch.hub.download_url_to_file(url, weights_path)
            else:
                raise ValueError(f"Cannot find weights path {weights_path}")
        detector = detector_type(model_path=weights_path, **detector_kwargs)

        # sanity check: selected detector can detect all classes in class_names
        detector_class_names = set(detector.cls_names)
        assert detector_class_names.issuperset(self.class_names), \
            f"Sanity check failed: class names of {self.class_names} can not "
        f"be detected by {detector_name}."

        return detector

    def _gen_vobj_data(self, frame_image):
        detector_outputs = self.detector.inference(frame_image)
        vobj_data = defaultdict(list)
        for d in detector_outputs:
            class_name = self.detector.cls_names[d["class_id"]]
            if class_name in self.class_names:
                del d["class_id"]
                vobj_data[class_name].append(d)
        return vobj_data

    def next(self) -> Frame:
        if self.has_next():
            frame = self.prev.next()
            vobj_data = self._gen_vobj_data(frame.image)
            # Sanity check: the new detected classes don't exist in vobj_data.
            # Different detectors should not detect the same class.
            assert not self.class_names & frame.vobj_data.keys()
            frame.vobj_data.update(vobj_data)
            return frame
        else:
            raise StopIteration
