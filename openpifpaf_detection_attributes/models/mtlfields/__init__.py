import openpifpaf

from .basenetwork import ForkNormNetwork
from .decoder import InstanceDecoder
from .head import AttributeField
from .loss import AttributeLoss
from ...datasets import headmeta


def register():
    openpifpaf.BASE_TYPES.add(ForkNormNetwork)
    for backbone in list(openpifpaf.BASE_FACTORIES.keys()):
        openpifpaf.BASE_FACTORIES['fn-'+backbone] = (lambda backbone=backbone:
            ForkNormNetwork('fn-'+backbone, backbone))
    openpifpaf.HEADS[headmeta.AttributeMeta] = AttributeField
    openpifpaf.DECODERS.add(InstanceDecoder)
    openpifpaf.LOSSES[headmeta.AttributeMeta] = AttributeLoss
