from abc import abstractmethod
from typing import Dict

import openpifpaf

from .attribute import ObjectType


class AnnotationAttr(openpifpaf.annotation.Base):
    """Annotation class for a detected instance."""

    object_type = None
    attribute_metas = None


    def __init__(self, **kwargs):
        self.id = kwargs['id'] if 'id' in kwargs else None
        self.ignore_eval = kwargs['ignore_eval'] if 'ignore_eval' in kwargs else None
        self.attributes = {}
        for meta in self.attribute_metas:
            if meta['attribute'] in kwargs:
                self.attributes[meta['attribute']] = kwargs[meta['attribute']]


    @abstractmethod
    def inverse_transform(self, meta):
        """Inverse data augmentation to get annotations on original images.
        Needs to be implemented for every type of object.
        """
        raise NotImplementedError


    def json_data(self):
        return {'object_type': self.object_type.name, **self.attributes}


"""List of annotations for every dataset and object type."""
OBJECT_ANNOTATIONS: Dict[str, Dict[ObjectType, AnnotationAttr]] = {}
