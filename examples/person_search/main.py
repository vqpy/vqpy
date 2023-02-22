import argparse
import os

import numpy as np

import vqpy


def make_parser():
    parser = argparse.ArgumentParser('VQPy Demo!')
    parser.add_argument('--path', help='path to video')
    parser.add_argument(
        "--save_folder",
        default=None,
        help="the folder to save the final result",
    )
    parser.add_argument(
        "--query_folder",
        default=None,
        help="the folder containing query images"
    )
    return parser


class Person(vqpy.VObjBase):
    required_fields = ['class_id', 'image']

    # default values, to be assigned in main()
    feature_predictor = None
    gallery_features = None

    @vqpy.property()
    @vqpy.stateful(30)
    def feature(self):
        """
        extract the feature of person image
        :return: feature vector, shape = (N,)
        """
        image = self.getv('image')
        if image is None:
            return None
        return Person.feature_predictor(image).reshape(-1)

    @vqpy.property()
    def candidate(self):
        """
        retrieve the top-1 similar query object as the searching candidate
        :returns:
            ids (int): query IDs with most similarity
            dist (float): the similarity distance with [0, 1]
        """
        query_features = [self.getv('feature', (-1) * i) for i in range(1, 31)]
        gallery_features = self.getv('gallery_features')

        # compare the feature distance for different target person
        past_ids, past_dist = [], []
        for query_feature in query_features:
            # iterate features from the last 30 frames
            if query_feature is not None:
                dist = np.dot(gallery_features, query_feature)  # cosine similarity distance
                past_ids.append(np.argmax(dist))  # the most similar IDs
                past_dist.append(np.max(dist))  # the most similarity distance

        ids = np.argmax(np.bincount(past_ids))  # the most matched IDs
        dist = np.mean(past_dist)  # the mean distance over past matching

        return ids, dist


class PersonSearch(vqpy.QueryBase):
    """The class searching target person from videos"""

    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        filter_cons = {
            '__class__': lambda x: x == Person,
            'candidate': lambda x: x[1] >= 0.97,  # similar threshold
        }

        select_cons = {
            'track_id': None,
            'candidate': lambda x: str(x[0]),  # convert IDs to string
                                               # for JSON serialization
            'tlbr': lambda x: str(x),  # convert to string
                                       # for JSON serialization
        }

        return vqpy.VObjConstraint(filter_cons=filter_cons,
                                   select_cons=select_cons,
                                   filename='person_search')


if __name__ == '__main__':
    args = make_parser().parse_args()

    # load pre-trained models for person feature extracting
    from models import ReIDPredictor
    feature_predictor = ReIDPredictor(cfg="MSMT17/bagtricks_R50.yml") # https://github.com/JDAI-CV/fast-reid/tree/master/configs

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

    vqpy.launch(cls_name=vqpy.COCO_CLASSES,
                cls_type={"person": Person},
                tasks=[PersonSearch()],
                video_path=args.path,
                save_folder=args.save_folder,
                )
