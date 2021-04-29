import argparse

import torch
import openpifpaf

from .attribute import JaadType
from .dataset import JaadDataset
from . import transforms
from .. import annotation
from .. import attribute
from .. import encoder
from .. import headmeta
from .. import metrics as eval_metrics
from .. import sampler


class Jaad(openpifpaf.datasets.DataModule):
    """DataModule for dataset JAAD."""

    debug = False
    pin_memory = False

    # General
    root_dir = 'data-jaad/'
    subset = 'default'
    train_set = 'train'
    val_set = 'val'
    test_set = 'test'
    subepochs = 1

    # Tasks
    pedestrian_attributes = ['detection']
    occlusion_level = 1
    upsample_stride = 1

    # Pre-processing
    image_width = 961
    top_crop_ratio = 0.33
    image_height_stride = 16
    fast_scaling = True
    augmentation = True


    def __init__(self):
        super().__init__()
        self.compute_attributes()
        self.compute_head_metas()


    @classmethod
    def compute_attributes(cls):
        cls.attributes = {
            JaadType.PEDESTRIAN: cls.pedestrian_attributes,
        }


    @classmethod
    def compute_head_metas(cls):
        att_metas = attribute.get_attribute_metas(dataset='jaad',
                                                  attributes=cls.attributes)
        cls.head_metas = [headmeta.AttributeMeta('attribute-'+am['attribute'],
                                                 'jaad', **am)
                          for am in att_metas]
        for hm in cls.head_metas:
            hm.upsample_stride = cls.upsample_stride


    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module Jaad')

        # General
        group.add_argument('--jaad-root-dir',
                           default=cls.root_dir,
                           help='root directory of jaad dataset')
        group.add_argument('--jaad-subset',
                           default=cls.subset,
                           choices=['default', 'all_videos', 'high_visibility'],
                           help='subset of videos to consider')
        group.add_argument('--jaad-training-set',
                           default=cls.train_set,
                           choices=['train', 'trainval'],
                           help='training set')
        group.add_argument('--jaad-validation-set',
                           default=cls.val_set,
                           choices=['val', 'test'],
                           help='validation set')
        group.add_argument('--jaad-testing-set',
                           default=cls.test_set,
                           choices=['val', 'test'],
                           help='testing set')
        group.add_argument('--jaad-subepochs',
                           default=cls.subepochs, type=int,
                           help='number of subepochs with sub-sampling')

        # Tasks
        group.add_argument('--jaad-pedestrian-attributes',
                           default=cls.pedestrian_attributes, nargs='+',
                           help='list of attributes to consider for pedestrians')
        group.add_argument('--jaad-occlusion-level',
                           default=cls.occlusion_level, type=int,
                           choices=[0, 1, 2],
                           help='max level of occlusion to learn from')
        group.add_argument('--jaad-head-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')

        # Pre-processing
        group.add_argument('--jaad-image-width',
                           default=cls.image_width, type=int,
                           help='width to rescale image to')
        group.add_argument('--jaad-top-crop-ratio',
                           default=cls.top_crop_ratio, type=float,
                           help='ratio of height to crop from top of image')
        group.add_argument('--jaad-image-height-stride',
                           default=cls.image_height_stride, type=int,
                           help='stride to compute height of image')
        assert cls.fast_scaling
        group.add_argument('--jaad-no-fast-scaling',
                           dest='jaad_fast_scaling',
                           default=True, action='store_false',
                           help='do not use fast scaling algorithm')
        assert cls.augmentation
        group.add_argument('--jaad-no-augmentation',
                           dest='jaad_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')


    @classmethod
    def configure(cls, args: argparse.Namespace):
        # Extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # General
        cls.root_dir = args.jaad_root_dir
        cls.subset = args.jaad_subset
        cls.train_set = args.jaad_training_set
        cls.val_set = args.jaad_validation_set
        cls.test_set = args.jaad_testing_set
        cls.subepochs = args.jaad_subepochs

        # Tasks
        cls.pedestrian_attributes = args.jaad_pedestrian_attributes
        cls.compute_attributes()
        cls.occlusion_level = args.jaad_occlusion_level
        cls.upsample_stride = args.jaad_head_upsample
        cls.compute_head_metas()

        # Pre-processing
        cls.image_width = args.jaad_image_width
        cls.top_crop_ratio = args.jaad_top_crop_ratio
        cls.image_height_stride = args.jaad_image_height_stride
        cls.fast_scaling = args.jaad_fast_scaling
        cls.augmentation = args.jaad_augmentation


    def _common_preprocess_op(self):
        return [
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(self.image_width,
                                       fast=self.fast_scaling),
            transforms.CropTopOut(self.top_crop_ratio,
                                  self.image_height_stride),
        ]


    def _train_preprocess(self):
        if self.augmentation:
            data_augmentation_op = [
                transforms.ZoomInOrOut(fast=self.fast_scaling),
                openpifpaf.transforms.RandomApply(transforms.HFlip(), 0.5),
                transforms.TRAIN_TRANSFORM,
            ]
        else:
            data_augmentation_op = [transforms.EVAL_TRANSFORM]

        encoders = [encoder.AttributeEncoder(
                        head_meta,
                        occlusion_level=self.occlusion_level,
                    )
                    for head_meta in self.head_metas]

        return openpifpaf.transforms.Compose([
            *self._common_preprocess_op(),
            *data_augmentation_op,
            openpifpaf.transforms.Encoders(encoders),
        ])


    def _eval_preprocess(self):
        return openpifpaf.transforms.Compose([
            *self._common_preprocess_op(),
            transforms.ToAnnotations(annotation.OBJECT_ANNOTATIONS['jaad']),
            transforms.EVAL_TRANSFORM,
        ])


    def train_loader(self):
        train_data = JaadDataset(
            root_dir=self.root_dir,
            split=self.train_set,
            subset=self.subset,
            preprocess=self._train_preprocess(),
        )
        subsampler = sampler.RegularSubSampler(
            len(train_data),
            subepochs=self.subepochs,
            shuffle=(not self.debug) and self.augmentation
        )
        return torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            sampler=subsampler,
            pin_memory=self.pin_memory,
            num_workers=self.loader_workers,
            drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta,
        )


    def val_loader(self):
        val_data = JaadDataset(
            root_dir=self.root_dir,
            split=self.val_set,
            subset=self.subset,
            preprocess=self._train_preprocess(),
        )
        return torch.utils.data.DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=(not self.debug) and self.augmentation,
            pin_memory=self.pin_memory,
            num_workers=self.loader_workers,
            drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta,
        )


    def eval_loader(self):
        eval_data = JaadDataset(
            root_dir=self.root_dir,
            split=self.test_set,
            subset=self.subset,
            preprocess=self._eval_preprocess(),
        )
        return torch.utils.data.DataLoader(
            eval_data,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.loader_workers,
            drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta,
        )


    def metrics(self):
        return [eval_metrics.InstanceDetection(self.head_metas)]
