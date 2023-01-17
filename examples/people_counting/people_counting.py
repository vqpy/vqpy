import argparse
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
        "-d",
        "--pretrained_model_dir",
        help="Directory to pretrained models")
    return parser


class Person(vqpy.VObjBase):

    @vqpy.property()
    @vqpy.stateful(4)
    def direction_vector(self):
        tlbr_c, tlbr_p = self.getv('tlbr'), self.getv('tlbr', -5)
        if tlbr_c is None or tlbr_p is None:
            return None
        center_c = (tlbr_c[:2] + tlbr_c[2:]) / 2
        center_p = (tlbr_p[:2] + tlbr_p[2:]) / 2
        diff = center_c - center_p
        return int(diff[0]), int(diff[1])

    @vqpy.property()
    @vqpy.stateful(4)
    def direction(self):
        def denoise(target, reference):
            THRESHOLD = 10
            if target != 0 and reference / target >= THRESHOLD:
                target = 0
            return target

        def get_name(value, pos_name, neg_name):
            if value > 0:
                result = pos_name
            elif value < 0:
                result = neg_name
            else:
                result = ""
            return result

        def get_center(tlbr):
            return (tlbr[:2] + tlbr[2:]) / 2

        def most_frequent(List):
            from collections import Counter
            occurence_count = Counter(List)
            return occurence_count.most_common(1)[0][0]

        hist_len = 5
        tlbr_past = [self.getv("tlbr", (-1)*i) for i in range(1, 1 + hist_len)]
        for value in tlbr_past:
            if value is None:
                return None

        centers = list(map(get_center, tlbr_past))
        diffs = [centers[i+1] - centers[i] for i in range(hist_len - 1)]

        diff_xs = [denoise(diff[0], diff[1]) for diff in diffs]
        diff_ys = [denoise(diff[1], diff[0]) for diff in diffs]

        horizontal = most_frequent([get_name(diff_x, "right", "left")
                                    for diff_x in diff_xs])
        vertical = most_frequent([get_name(diff_y, "bottom", "top")
                                  for diff_y in diff_ys])
        direction = vertical + horizontal
        if direction == "":
            direction = None

        return direction


class CountPersonOnCrosswalk(vqpy.QueryBase):

    @staticmethod
    def set_output_configs() -> vqpy.OutputConfig:
        return vqpy.OutputConfig(
            # output_frame_vobj_num=True,
            output_total_vobj_num=True
            )

    @staticmethod
    def setting() -> vqpy.VObjConstraint:

        CROSSWALK_REGION_1 = [(731, 554), (963, 564), (436, 1076), (14, 1076)]
        CROSSWALK_REGION_2 = [(1250, 528), (1292, 473),
                              (1839, 492), (1893, 547)]
        CROSSWALK_REGIONS = [CROSSWALK_REGION_1, CROSSWALK_REGION_2]

        filter_cons = {"__class__": lambda x: x == Person,
                       "bottom_center": vqpy.utils.within_regions(
                            CROSSWALK_REGIONS
                            )}
        select_cons = {"track_id": None,
                       }
        return vqpy.VObjConstraint(filter_cons=filter_cons,
                                   select_cons=select_cons,
                                   filename='on_crosswalk')


class CountPersonHeadLeft(CountPersonOnCrosswalk):

    @staticmethod
    def set_output_configs() -> vqpy.OutputConfig:
        return vqpy.OutputConfig(
            # output_frame_vobj_num=True,
            output_total_vobj_num=True
            )

    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        filter_cons = {"direction": lambda x: "left" in x}
        select_cons = {"track_id": None,
                       "direction": None,
                       "direction_vector": None
                       }
        return vqpy.VObjConstraint(filter_cons=filter_cons,
                                   select_cons=select_cons,
                                   filename='head_left')


class CountPersonHeadRight(CountPersonOnCrosswalk):

    @staticmethod
    def set_output_configs() -> vqpy.OutputConfig:
        return vqpy.OutputConfig(
            # output_frame_vobj_num=True,
            output_total_vobj_num=True
            )

    @staticmethod
    def setting() -> vqpy.VObjConstraint:
        filter_cons = {"direction": lambda x: "right" in x}
        select_cons = {"track_id": None,
                       "direction": None,
                       "direction_vector": None
                       }
        return vqpy.VObjConstraint(filter_cons=filter_cons,
                                   select_cons=select_cons,
                                   filename='head_right')


if __name__ == '__main__':
    args = make_parser().parse_args()
    vqpy.launch(cls_name=vqpy.COCO_CLASSES,
                cls_type={"person": Person},
                tasks=[CountPersonHeadLeft(), CountPersonHeadRight()],
                video_path=args.path,
                save_folder=args.save_folder,
                detector_model_dir=args.pretrained_model_dir)
