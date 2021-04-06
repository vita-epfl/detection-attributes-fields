import copy

from .attribute import JaadType, JAAD_ATTRIBUTE_METAS
from .. import prediction


class JaadPedestrianPrediction(prediction.Prediction):
    object_type = JaadType.PEDESTRIAN
    attribute_metas = JAAD_ATTRIBUTE_METAS[JaadType.PEDESTRIAN]


    def inverse_transform(self, meta):
        pred = copy.deepcopy(self)

        # Horizontal flip
        if meta['hflip']:
            w = meta['width_height'][0]
            if pred['center'] is not None:
                pred['center'][0] = -pred['center'][0] + (w - 1)
            pred['bag_left_side'], pred['bag_right_side'] = (
                pred['bag_right_side'], pred['bag_left_side'])
            pred['pose_left'], pred['pose_right'] = (
                pred['pose_right'], pred['pose_left'])

        # Offset and scale
        if pred['center'] is not None:
            pred['center'][0] = (pred['center'][0] + meta['offset'][0]) / meta['scale'][0]
            pred['center'][1] = (pred['center'][1] + meta['offset'][1]) / meta['scale'][1]
        if pred['width'] is not None:
            pred['width'] /= meta['scale'][0]
        if pred['height'] is not None:
            pred['height'] /= meta['scale'][1]

        return pred


JAAD_OBJECT_PREDICTIONS = {
    JaadType.PEDESTRIAN: JaadPedestrianPrediction,
}
