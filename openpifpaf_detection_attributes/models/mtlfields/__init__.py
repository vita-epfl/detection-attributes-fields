import openpifpaf

from .basenetwork import ForkNormNetwork
from .decoder import InstanceDecoder
from .head import AttributeField
from .loss import AttributeLoss
from ...datasets import headmeta


def register():
    openpifpaf.BASE_TYPES.add(ForkNormNetwork)
    openpifpaf.BASE_FACTORIES['fn-resnet50'] = lambda: ForkNormNetwork(
        'fn-resnet50', 'resnet50')
    openpifpaf.HEADS[headmeta.AttributeMeta] = AttributeField
    openpifpaf.DECODERS.add(InstanceDecoder)
    openpifpaf.LOSSES[headmeta.AttributeMeta] = AttributeLoss
