import openpifpaf

from .annotation import JAAD_OBJECT_ANNOTATIONS
from .attribute import JaadType, JAAD_ATTRIBUTE_METAS
from .datamodule import Jaad
from .encoder import JAAD_ATTRIBUTE_GENERATORS
from .. import annotation
from .. import attribute
from .. import encoder
from .. import painter


def register():
    openpifpaf.DATAMODULES['jaad'] = Jaad
    openpifpaf.PAINTERS['JaadPedestrianAnnotation'] = painter.BoxPainter

    attribute.OBJECT_TYPES['jaad'] = JaadType
    attribute.ATTRIBUTE_METAS['jaad'] = JAAD_ATTRIBUTE_METAS
    encoder.ATTRIBUTE_GENERATORS['jaad'] = JAAD_ATTRIBUTE_GENERATORS
    annotation.OBJECT_ANNOTATIONS['jaad'] = JAAD_OBJECT_ANNOTATIONS
