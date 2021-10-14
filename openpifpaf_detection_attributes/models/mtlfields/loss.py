import argparse
import logging

import torch

from ...datasets import headmeta


LOG = logging.getLogger(__name__)


class AttributeLoss(torch.nn.Module):
    """Loss function for attribute fields.

    Args:
        head_meta (AttributeMeta): Meta information on attribute to predict.
    """

    regression_loss = 'l1'
    focal_gamma = 0.0


    def __init__(self, head_meta: headmeta.AttributeMeta):
        super().__init__()
        self.meta = head_meta
        self.field_names = ['{}.{}'.format(head_meta.dataset,
                                           head_meta.name)]
        self.previous_loss = None

        LOG.debug('attribute loss for %s: %s, %d channels',
                  self.meta.attribute,
                  ('classification' if self.meta.is_classification
                   else 'regression'),
                  self.meta.n_channels)


    @property
    def loss_function(self):
        if self.meta.is_classification:
            if self.meta.n_channels == 1:
                return torch.nn.BCEWithLogitsLoss(reduction='none')
            elif self.meta.n_channels > 1:
                loss_module = torch.nn.CrossEntropyLoss(reduction='none')
                return lambda x, t: loss_module(
                        x, t.to(torch.long).squeeze(1)).unsqueeze(1)
            else:
                raise Exception('error in attribute classification format:'
                                ' size {}'.format(self.meta.n_channels))
        else:
            if self.regression_loss == 'l1':
                return torch.nn.L1Loss(reduction='none')
            elif self.regression_loss == 'l2':
                return torch.nn.MSELoss(reduction='none')
            elif self.regression_loss == 'smoothl1':
                return torch.nn.SmoothL1Loss(reduction='none')
            else:
                raise Exception('unknown attribute regression loss type {}'
                                ''.format(self.regression_loss))


    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('AttributeLoss')
        group.add_argument('--attribute-regression-loss',
                           default=cls.regression_loss,
                           choices=['l1', 'l2', 'smoothl1'],
                           help='type of regression loss for attributes')
        group.add_argument('--attribute-focal-gamma',
                           default=cls.focal_gamma, type=float,
                           help='use focal loss for attributes with the given'
                                ' gamma')


    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.regression_loss = args.attribute_regression_loss
        cls.focal_gamma = args.attribute_focal_gamma


    def forward(self, *args):
        LOG.debug('loss for %s', self.field_names)

        x, t = args
        loss = self.compute_loss(x, t)

        if (loss is not None) and (not torch.isfinite(loss).item()):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(loss, self.previous_loss))
        self.previous_loss = float(loss.item()) if loss is not None else None

        return [loss]


    def compute_loss(self, x, t):
        if t is None:
            return None

        c_x = x.shape[1]
        x = x.permute(0,2,3,1).reshape(-1, c_x)
        c_t = t.shape[1]
        t = t.permute(0,2,3,1).reshape(-1, c_t)

        mask = torch.isnan(t).any(1).bitwise_not_()
        if not torch.any(mask):
            return None

        x = x[mask, :]
        t = t[mask, :]
        loss = self.loss_function(x, t)

        if (self.focal_gamma != 0) and self.meta.is_classification:
            if self.meta.n_channels == 1: # BCE
                focal = torch.sigmoid(x)
                focal = torch.where(t < 0.5, focal, 1. - focal)
            else: # CE
                focal = torch.nn.functional.softmax(x, dim=1)
                focal = 1. - focal.gather(1, t.to(torch.long))
            loss = loss * focal.pow(self.focal_gamma)

        loss = loss.mean()
        return loss
