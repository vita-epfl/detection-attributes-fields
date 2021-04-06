from abc import abstractmethod
from typing import Dict

import openpifpaf

from .attribute import ObjectType


class Prediction(openpifpaf.annotation.Base):
    object_type = None
    attribute_metas = None


    def __init__(self, **kwargs):
        self.attributes = {}
        for meta in self.attribute_metas:
            if meta.attribute in kwargs:
                self.attributes[meta.attribute] = kwargs[meta.attribute]
            else:
                self.attributes[meta.attribute] = None


    @abstractmethod
    def inverse_transform(self, meta):
        raise NotImplementedError


    def json_data(self):
        return {'object_type': self.object_type, **self.attributes}


OBJECT_PREDICTIONS: Dict[str, Dict[ObjectType, Prediction]] = {}
