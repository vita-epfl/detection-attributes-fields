import argparse
import logging

import openpifpaf

from .. import mtl_grad_fork_norm


LOG = logging.getLogger(__name__)


class ForkNormNetwork(openpifpaf.network.basenetworks.BaseNetwork):
    """Backbone network with fork-normalization before prediction head
        networks.

    Args:
        name (str): Name of network.
        backbone_name (str): Name of base network (without fork_normalization).
    """

    pifpaf_pretraining = False
    fork_normalization_operation = 'accumulation'
    fork_normalization_duplicates = 1


    def __init__(self, name: str, backbone_name: str):
        if self.pifpaf_pretraining:
            # Load pre-trained weights
            LOG.info('Loading weights from OpenPifPaf trained model')
            network_factory = openpifpaf.network.Factory()
            network_factory.checkpoint = backbone_name
            pretrained_net, _ = network_factory.from_checkpoint()
            backbone = pretrained_net.base_net
        else:
            # Build from scratch
            backbone = openpifpaf.BASE_FACTORIES[backbone_name]()
        super().__init__(name,
                         stride=backbone.stride,
                         out_features=backbone.out_features)
        self.backbone_name = backbone_name
        self.backbone = backbone
        self.fork_normalization = mtl_grad_fork_norm.MtlGradForkNorm(
            normalization=self.fork_normalization_operation,
            duplicates=self.fork_normalization_duplicates,
        )


    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Fork-Normalized Network')
        group.add_argument('--pifpaf-pretraining',
                           dest='pifpaf_pretraining', action='store_true',
                           default=False,
                           help='initialization from PifPaf pretrained model')
        group.add_argument('--fork-normalization-operation',
                           default=cls.fork_normalization_operation,
                           choices=['accumulation', 'average', 'power',
                                    'sample', 'random'],
                           help='operation for fork-normalization')
        group.add_argument('--fork-normalization-duplicates',
                           default=cls.fork_normalization_duplicates, type=int,
                           help='max number of branches to fork-normalize for')


    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.pifpaf_pretraining = args.pifpaf_pretraining
        cls.fork_normalization_operation = args.fork_normalization_operation
        cls.fork_normalization_duplicates = args.fork_normalization_duplicates


    def forward(self, *args):
        x = args[0]
        x = self.backbone(x)
        x = self.fork_normalization(x)
        return x
