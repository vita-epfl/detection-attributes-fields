from abc import abstractmethod
from enum import auto, Enum
import logging
from typing import Dict

from .attribute import ObjectType
from .headmeta import AttributeMeta


LOG = logging.getLogger(__name__)


class AnnotationRescaler:
    def __init__(self, stride, object_type):
        self.stride = stride
        self.object_type = object_type


    def valid_area(self, meta):
        if 'valid_area' not in meta:
            return None

        return (
            meta['valid_area'][0] / self.stride,
            meta['valid_area'][1] / self.stride,
            meta['valid_area'][2] / self.stride,
            meta['valid_area'][3] / self.stride,
        )


    @abstractmethod
    def objects(self, anns):
        raise NotImplementedError


    def width_height(self, width_height_original):
        return [round((width_height_original[0]-1) / self.stride + 1),
                round((width_height_original[1]-1) / self.stride + 1)]


class AttributeEncoder:
    def __init__(self,
                 meta: AttributeMeta,
                 rescaler: AnnotationRescaler = None,
                 **kwargs):
        self.meta = meta
        self.rescaler = rescaler
        self.__dict__.update(kwargs)


    def __call__(self, image, anns, meta):
        generator = ATTRIBUTE_GENERATORS[self.meta.dataset][self.meta.object_type]
        return generator(self)(image, anns, meta)


class AttributeGenerator:
    rescaler_class = AnnotationRescaler


    def __init__(self, config: AttributeEncoder):
        self.config = config
        self.rescaler = config.rescaler or self.rescaler_class(
            config.meta.stride, config.meta.object_type)


    def __call__(self, image, anns, meta):
        width_height_original = image.shape[2:0:-1]

        objects = self.rescaler.objects(anns)
        new_width_height = self.rescaler.width_height(width_height_original)
        valid_area = self.rescaler.valid_area(meta)
        LOG.debug('valid area: %s', valid_area)

        encoding = self.generate_encoding(objects, new_width_height, valid_area)
        return encoding


    @abstractmethod
    def generate_encoding(self, objects, width_height, valid_area):
        raise NotImplementedError


ATTRIBUTE_GENERATORS: Dict[str, Dict[ObjectType, AttributeGenerator]] = {}
