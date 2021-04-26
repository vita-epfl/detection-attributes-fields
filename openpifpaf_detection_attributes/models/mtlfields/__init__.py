import openpifpaf

from .basenetwork import ForkNormNetwork
from .decoder import InstanceDecoder
from .head import AttributeField
from .loss import AttributeLoss
from ...datasets import headmeta


def register():
    openpifpaf.BASE_TYPES.add(ForkNormNetwork)
    openpifpaf.BASE_FACTORIES['fn-mobilenetv2'] = lambda: ForkNormNetwork(
        'fn-mobilenetv2', 'mobilenetv2')
    openpifpaf.BASE_FACTORIES['fn-resnet50'] = lambda: ForkNormNetwork(
        'fn-resnet50', 'resnet50')
    openpifpaf.BASE_FACTORIES['fn-shufflenetv2k16'] = lambda: ForkNormNetwork(
        'fn-shufflenetv2k16', 'shufflenetv2k16')
    openpifpaf.BASE_FACTORIES['fn-shufflenetv2k30'] = lambda: ForkNormNetwork(
        'fn-shufflenetv2k30', 'shufflenetv2k30')
    openpifpaf.HEADS[headmeta.AttributeMeta] = AttributeField
    openpifpaf.DECODERS.add(InstanceDecoder)
    openpifpaf.LOSSES[headmeta.AttributeMeta] = AttributeLoss
