import argparse
import logging
import math

import openpifpaf
import torch

from ...datasets import headmeta


LOG = logging.getLogger(__name__)


class AttributeField(openpifpaf.network.heads.HeadNetwork):
    """Pediction head network for attributes.

    Args:
        meta (AttributeMeta): Meta information on attribute to predict.
        in_features (int): Number of features as input to the head network.
    """

    # Convolutions
    detection_bias_prior = None


    def __init__(self,
                 meta: headmeta.AttributeMeta,
                 in_features: int):
        super().__init__(meta, in_features)

        LOG.debug('%s config: dataset %s, attribute %s',
                  meta.name, meta.dataset, meta.attribute)

        # Convolutions
        out_features = meta.n_channels * meta.upsample_stride**2
        self.conv = torch.nn.Conv2d(in_features, out_features,
                                    kernel_size=1, padding=0, dilation=1)
        if (
            (self.detection_bias_prior is not None)
            and (meta.attribute == 'confidence')
        ):
            assert (
                (self.detection_bias_prior > 0.)
                and (self.detection_bias_prior < 1.)
            )
            self.conv.bias.data.fill_(-math.log(
                (1. - self.detection_bias_prior) / self.detection_bias_prior))

        # Upsampling
        assert meta.upsample_stride >= 1
        self.upsample_op = None
        if meta.upsample_stride > 1:
            self.upsample_op = torch.nn.PixelShuffle(meta.upsample_stride)


    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('AttributeField')

        # Convolutions
        group.add_argument('--detection-bias-prior',
                           default=cls.detection_bias_prior, type=float,
                           help='prior bias for detection')


    @classmethod
    def configure(cls, args: argparse.Namespace):
        # Convolutions
        cls.detection_bias_prior = args.detection_bias_prior


    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[self.meta.head_index]
        x = self.conv(x)

        # Upsampling
        if self.upsample_op is not None:
            x = self.upsample_op(x)
            low_cut = (self.meta.upsample_stride - 1) // 2
            high_cut = math.ceil((self.meta.upsample_stride - 1) / 2.0)
            if self.training:
                # Negative axes not supported by ONNX TensorRT
                x = x[:, :, low_cut:-high_cut, low_cut:-high_cut]
            else:
                # The int() forces the tracer to use static shape
                x = x[:, :,
                      low_cut:int(x.shape[2]) - high_cut,
                      low_cut:int(x.shape[3]) - high_cut]

        return x
