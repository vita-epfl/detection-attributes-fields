import openpifpaf

from . import datamodule
from .attribute import JaadType, JAAD_ATTRIBUTE_METAS
from .encoder import JAAD_ATTRIBUTE_GENERATORS
from .prediction import JAAD_OBJECT_PREDICTIONS
from .. import attribute, encoder, prediction


def register():
    openpifpaf.DATAMODULES['jaad'] = datamodule.Jaad

    attribute.OBJECT_TYPES['jaad'] = JaadType
    attribute.ATTRIBUTE_METAS['jaad'] = JAAD_ATTRIBUTE_METAS
    encoder.ATTRIBUTE_GENERATORS['jaad'] = JAAD_ATTRIBUTE_GENERATORS
    prediction.OBJECT_PREDICTIONS['jaad'] = JAAD_OBJECT_PREDICTIONS
