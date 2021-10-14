import argparse
import logging
import time
from typing import List

import numpy as np
import openpifpaf
from scipy.special import softmax
import torch

from .. import optics
from ...datasets import annotation
from ...datasets import attribute
from ...datasets import headmeta


LOG = logging.getLogger(__name__)


class InstanceDecoder(openpifpaf.decoder.decoder.Decoder):
    """Decoder to convert predicted fields to sets of instance detections.

    Args:
        dataset (str): Dataset name.
        object_type (ObjectType): Type of object detected.
        attribute_metas (List[AttributeMeta]): List of meta information about
            predicted attributes.
    """

    # General
    dataset = None
    object_type = None

    # Clustering detections
    s_threshold = 0.2
    optics_min_cluster_size = 10
    optics_epsilon = 5.0
    optics_cluster_threshold = 0.5


    def __init__(self,
                 dataset: str,
                 object_type: attribute.ObjectType,
                 attribute_metas: List[headmeta.AttributeMeta]):
        super().__init__()
        self.dataset = dataset
        self.object_type = object_type
        self.annotation = annotation.OBJECT_ANNOTATIONS[self.dataset][self.object_type]
        for meta in attribute_metas:
            assert meta.dataset == self.dataset
            assert meta.object_type is self.object_type
        self.attribute_metas = attribute_metas


    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('InstanceDecoder')

        # Clustering detections
        group.add_argument('--decoder-s-threshold',
                           default=cls.s_threshold, type=float,
                           help='threshold for field S')
        group.add_argument('--decoder-optics-min-cluster-size',
                           default=cls.optics_min_cluster_size, type=int,
                           help='minimum size of clusters in OPTICS')
        group.add_argument('--decoder-optics-epsilon',
                           default=cls.optics_epsilon, type=float,
                           help='maximum radius of cluster in OPTICS')
        group.add_argument('--decoder-optics-cluster-threshold',
                           default=cls.optics_cluster_threshold, type=float,
                           help='threshold to separate clusters in OPTICS')


    @classmethod
    def configure(cls, args: argparse.Namespace):
        # Clustering detections
        cls.s_threshold = args.decoder_s_threshold
        cls.optics_min_cluster_size = args.decoder_optics_min_cluster_size
        cls.optics_epsilon = args.decoder_optics_epsilon
        cls.optics_cluster_threshold = args.decoder_optics_cluster_threshold


    @classmethod
    def factory(self, head_metas: List[openpifpaf.headmeta.Base]):
        decoders = []
        for dataset in attribute.OBJECT_TYPES:
            for object_type in attribute.OBJECT_TYPES[dataset]:
                meta_list = [meta for meta in head_metas
                             if (
                                isinstance(meta, headmeta.AttributeMeta)
                                and (meta.dataset == dataset)
                                and (meta.object_type is object_type)
                             )]
                if len(meta_list) > 0:
                    decoders.append(InstanceDecoder(dataset=dataset,
                                                    object_type=object_type,
                                                    attribute_metas=meta_list))
        return decoders


    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()

        # Conversion to numpy if needed
        fields = [f.numpy() if torch.is_tensor(f) else f for f in fields]

        # Field S
        s_meta = [meta for meta in self.attribute_metas
                  if meta.attribute == 'confidence']
        assert len(s_meta) == 1
        s_meta = s_meta[0]
        s_field = fields[s_meta.head_index].copy()
        conf_field = 1. / (1. + np.exp(-s_field))
        s_mask = conf_field > self.s_threshold

        # Field V
        v_meta = [meta for meta in self.attribute_metas
                  if meta.attribute == 'center']
        assert len(v_meta) == 1
        v_meta = v_meta[0]
        v_field = fields[v_meta.head_index].copy()
        if v_meta.std is not None:
            v_field[0] *= v_meta.std[0]
            v_field[1] *= v_meta.std[1]
        if v_meta.mean is not None:
            v_field[0] += v_meta.mean[0]
            v_field[1] += v_meta.mean[1]

        # OPTICS clustering
        point_list = []
        for y in range(s_mask.shape[1]):
            for x in range(s_mask.shape[2]):
                if s_mask[0,y,x]:
                    point = optics.Point(x, y, v_field[0,y,x], v_field[1,y,x])
                    point_list.append(point)

        clustering = optics.Optics(point_list,
                                   self.optics_min_cluster_size,
                                   self.optics_epsilon)
        clustering.run()
        clusters = clustering.cluster(self.optics_cluster_threshold)

        # Predictions for all instances
        predictions = []
        for cluster in clusters:
            attributes = {}
            for meta in self.attribute_metas:
                att = self.cluster_vote(fields[meta.head_index], cluster,
                                        meta, conf_field)
                attributes[meta.attribute] = att

            pred = self.annotation(**attributes)
            predictions.append(pred)

        LOG.info('predictions %d, %.3fs',
                  len(predictions), time.perf_counter()-start)

        return predictions


    def cluster_vote(self, field, cluster, meta, conf_field):
        field = field.copy()

        if meta.std is not None:
            field *= (meta.std if meta.n_channels == 1
                      else np.expand_dims(meta.std, (1,2)))
        if meta.mean is not None:
            field += (meta.mean if meta.n_channels == 1
                      else np.expand_dims(meta.mean, (1,2)))

        pred = np.array([0.]*field.shape[0])
        norm = 0.
        for pt in cluster.points:
            if meta.is_scalar: # scalar field
                val = field[:, pt.y, pt.x]
            else: # vectorial field
                val = np.array([pt.x, pt.y]) + field[:, pt.y, pt.x]
            conf = (
                conf_field[0, pt.y, pt.x] if meta.attribute != 'confidence'
                else 1.
            )
            pred += val * conf
            norm += conf
        pred = pred / norm if norm != 0. else 0.

        if meta.is_spatial:
            pred *= meta.stride
        if meta.n_channels == 1:
            if meta.is_classification:
                pred = 1. / (1. + np.exp(-pred))
            pred = pred[0]
        else:
            if meta.is_classification:
                pred = softmax(pred)
            pred = pred.tolist()

        return pred
