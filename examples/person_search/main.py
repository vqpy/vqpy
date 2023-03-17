import argparse
import os

import numpy as np

from vqpy.backend.plan import Planner, Executor
from vqpy.frontend.vobj import VObjBase, vobj_property
from vqpy.frontend.query import QueryBase


def make_parser():
    parser = argparse.ArgumentParser("VQPy Demo!")
    parser.add_argument("--path", help="path to video")
    parser.add_argument(
        "--save_folder",
        default=None,
        help="the folder to save the final result",
    )
    parser.add_argument(
        "--query_folder",
        default=None,
        help="the folder containing query images",
    )
    return parser


class Person(VObjBase):
    feature_predictor = None
    gallery_features = None

    def __init__(self) -> None:
        self.class_name = "person"
        self.object_detector = "yolox"
        self.detector_kwargs = {"device": "gpu"}
        super().__init__()

    @vobj_property(inputs={"tlbr": 0})
    def center(self, values):
        tlbr = values["tlbr"]
        return (tlbr[:2] + tlbr[2:]) / 2

    @vobj_property(inputs={"image": 0})
    def feature(self, values):
        image = values["image"]
        return Person.feature_predictor(image).reshape(-1)

    # require the past 10 frames. was 30, but for some reason, we're not able
    # to find any Person VObj with 30 frames of history
    @vobj_property(inputs={"feature": 10 - 1})
    def candidate(self, values):
        """
        retrieve the top-1 similar query object as the searching candidate
        :returns:
            ids (int): query IDs with most similarity
            dist (float): the similarity distance with [0, 1]
        """
        query_features = values["feature"]  # exclude the current frame
        gallery_features = Person.gallery_features

        # compare the feature distance for different target person
        past_ids, past_dist = [], []
        for query_feature in query_features:
            # iterate over the past 30 frames
            if query_feature is not None:
                # cosine similarity distance
                dist = np.dot(gallery_features, query_feature)
                past_ids.append(np.argmax(dist))  # the most similar IDs
                past_dist.append(np.max(dist))  # the most similarity distance

        ids = np.argmax(np.bincount(past_ids))  # the most matched IDs
        dist = np.mean(past_dist)  # the mean distance over past matching
        return ids, dist


class PersonSearch(QueryBase):
    """The class searching target person from videos"""

    def __init__(self) -> None:
        self.person = Person()
        super().__init__()

    def frame_constraint(self):
        return self.person.candidate.cmp(lambda x: x != None and x[1] >= 0.97)

    def frame_output(self):
        return {
            "track_id": self.person.track_id,
            "candidate": self.person.candidate,
            "tlbr": self.person.tlbr,
        }


if __name__ == "__main__":
    args = make_parser().parse_args()

    # load pre-trained models for person feature extracting
    from models import ReIDPredictor

    # https://github.com/JDAI-CV/fast-reid/tree/master/configs
    feature_predictor = ReIDPredictor(cfg="MSMT17/bagtricks_R50.yml")

    # extract the feature of query images
    gallery_features = []
    for file_name in os.listdir(args.query_folder):
        # extract features for all images from given directory
        img_path = os.path.join(args.query_folder, file_name)
        preds = feature_predictor(img_path)
        gallery_features.append(preds)
    gallery_features = np.concatenate(gallery_features, axis=0)

    Person.feature_predictor = feature_predictor
    Person.gallery_features = gallery_features

    planner = Planner()
    launch_args = {"video_path": args.path}
    root_plan_node = planner.parse(PersonSearch())
    planner.print_plan(root_plan_node)
    executor = Executor(root_plan_node, launch_args)
    result = executor.execute()

    for frame in result:
        print(frame.id)
        for person_idx in frame.filtered_vobjs[0]["person"]:
            print(frame.vobj_data["person"][person_idx])
