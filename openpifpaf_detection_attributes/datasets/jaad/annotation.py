import copy

from .attribute import JaadType, JAAD_ATTRIBUTE_METAS
from .. import annotation


class JaadPedestrianAnnotation(annotation.AnnotationAttr):
    """Annotation class for pedestrians from dataset JAAD."""

    object_type = JaadType.PEDESTRIAN
    attribute_metas = JAAD_ATTRIBUTE_METAS[JaadType.PEDESTRIAN]


    def inverse_transform(self, meta):
        pred = copy.deepcopy(self)

        atts = pred.attributes

        # Horizontal flip
        if meta['hflip']:
            w = meta['width_height'][0]
            if atts['center'] is not None:
                atts['center'][0] = -atts['center'][0] + (w - 1)
            atts['bag_left_side'], atts['bag_right_side'] = (
                atts['bag_right_side'], atts['bag_left_side'])
            atts['pose_left'], atts['pose_right'] = (
                atts['pose_right'], atts['pose_left'])

        # Offset and scale
        if atts['center'] is not None:
            atts['center'][0] = (atts['center'][0] + meta['offset'][0]) / meta['scale'][0]
            atts['center'][1] = (atts['center'][1] + meta['offset'][1]) / meta['scale'][1]
        if atts['width'] is not None:
            atts['width'] /= meta['scale'][0]
        if atts['height'] is not None:
            atts['height'] /= meta['scale'][1]

        return pred


JAAD_OBJECT_ANNOTATIONS = {
    JaadType.PEDESTRIAN: JaadPedestrianAnnotation,
}
