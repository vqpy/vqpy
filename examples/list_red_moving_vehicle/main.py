import vqpy
from vqpy.frontend.vobj import VObjBase, vobj_property
from vqpy.frontend.query import QueryBase
import math
import argparse
from getcolor import get_color


class Car(VObjBase):
    def __init__(self):
        self.class_name = "car"
        self.object_detector = "yolox"
        self.detector_kwargs = {"device": "cpu"}
        super().__init__()

    @vobj_property(inputs={"image": 0})
    def color(self, values):
        image = values["image"]
        color = get_color(image)
        return color

    @vobj_property(inputs={"image": 0, "license_plate": 2})
    def license_plate(self, values):
        """
        Note that the built in openalpr model is not very accurate.
        You can change the model to a better one.
        """
        from vqpy.property_lib.vehicle.models.openalpr import GetLP
        image = values["image"]
        last_license_plate = values["license_plate"][-2]
        if last_license_plate is None:
            license_plate = GetLP(image)
        else:
            license_plate = last_license_plate
        return license_plate

    @vobj_property(inputs={"tlbr": 1})
    def velocity(self, values):
        fps = 24.0
        last_tlbr, tlbr = values["tlbr"]
        if last_tlbr is None or tlbr is None:
            return 0
        last_center = (last_tlbr[:2] + last_tlbr[2:]) / 2
        cur_center = (tlbr[:2] + tlbr[2:]) / 2
        tlbr_avg = (tlbr + last_tlbr) / 2
        scale = (tlbr_avg[3] - tlbr_avg[1]) / 1.5
        dcenter = (cur_center - last_center) / scale * fps
        return math.sqrt(sum(dcenter * dcenter))


class ListSpeedingCar(QueryBase):
    def __init__(self):
        self.car = Car()

    def frame_constraint(self):
        return (self.car.score > 0.6) \
               & (self.car.velocity > 1.0)

    def frame_output(self):
        return (
            self.car.track_id,
            self.car.color,
        )


class FindAmberAlertCar(ListSpeedingCar):
    def frame_constraint(self):
        return super().frame_constraint() \
               & (self.car.color.cmp(lambda color: "red" in color))

    def frame_output(self):
        return (self.car.track_id,
                self.car.license_plate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to video file",
                        default="./license-10s.mp4")
    parser.add_argument("--save_folder", help="path to save query result")
    args = parser.parse_args()
    query_executor = vqpy.init(
        video_path=args.path,
        query_obj=FindAmberAlertCar(),
        verbose=True,
    )
    vqpy.run(query_executor, args.save_folder)
