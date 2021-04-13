from abc import abstractmethod
from typing import Dict

import openpifpaf

from .attribute import ObjectType


class Prediction(openpifpaf.annotation.Base):
    """Prediction output for a detected instance."""

    object_type = None
    attribute_metas = None


    def __init__(self, **kwargs):
        self.attributes = {}
        for meta in self.attribute_metas:
            if meta['attribute'] in kwargs:
                self.attributes[meta['attribute']] = kwargs[meta['attribute']]


    @abstractmethod
    def inverse_transform(self, meta):
        """Inverse data augmentation to get predictions on original images.
        Needs to be implemented for every type of object.
        """
        raise NotImplementedError


    def json_data(self):
        return {'object_type': self.object_type, **self.attributes}


"""List of predictions for every dataset and object type."""
OBJECT_PREDICTIONS: Dict[str, Dict[ObjectType, Prediction]] = {}
