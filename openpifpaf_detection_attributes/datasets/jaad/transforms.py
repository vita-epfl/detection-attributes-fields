import copy
import logging
import warnings

import numpy as np
import openpifpaf
import PIL
import scipy.ndimage
import torch
import torchvision

from .attribute import JaadType
from .. import annotation


LOG = logging.getLogger(__name__)


def _scale(image, anns, meta, target_w, target_h, resample, *, fast=False):
    """target_w and target_h as integers
    Internally, resample in Pillow are aliases:
    PIL.Image.BILINEAR = 2
    PIL.Image.BICUBIC = 3
    """
    assert resample in (0, 2, 3)
    meta = copy.deepcopy(meta)
    anns = copy.deepcopy(anns)
    w, h = image.size

    # Scale image
    if fast:
        image = image.resize((target_w, target_h), resample)
    else:
        order = resample
        if order == 2:
            order = 1

        im_np = np.asarray(image)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            im_np = scipy.ndimage.zoom(im_np, (target_h / h, target_w / w, 1),
                                       order=order)
        image = PIL.Image.fromarray(im_np)

    LOG.debug('before resize = (%f, %f), after = %s', w, h, image.size)
    assert image.size[0] == target_w
    assert image.size[1] == target_h

    # Rescale annotations
    x_scale = (image.size[0] - 1) / (w - 1)
    y_scale = (image.size[1] - 1) / (h - 1)
    for ann in anns:
        if ann['object_type'] is JaadType.PEDESTRIAN:
            ann['box'][0] *= x_scale
            ann['box'][1] *= y_scale
            ann['box'][2] *= x_scale
            ann['box'][3] *= y_scale
            ann['center'][0] *= x_scale
            ann['center'][1] *= y_scale
            ann['width'] *= x_scale
            ann['height'] *= y_scale

    # Adjust meta
    scale_factors = np.array((x_scale, y_scale))
    LOG.debug('meta before resize: %s', meta)
    meta['offset'] *= scale_factors
    meta['scale'] *= scale_factors
    meta['valid_area'][:2] *= scale_factors
    meta['valid_area'][2:] *= scale_factors
    LOG.debug('meta after resize: %s', meta)

    return image, anns, meta


class NormalizeAnnotations(openpifpaf.transforms.Preprocess):
    @staticmethod
    def normalize_annotations(anns):
        anns = copy.deepcopy(anns)

        for ann in anns:
            if isinstance(ann, annotation.AnnotationAttr):
                # Already converted to an annotation type
                continue

            if ann['object_type'] is JaadType.PEDESTRIAN:
                ann['box'] = np.asarray(ann['box'], dtype=np.float32)
                ann['center'] = np.asarray(ann['center'], dtype=np.float32)

        return anns


    def __call__(self, image, anns, meta):
        anns = self.normalize_annotations(anns)
        if meta is None:
            meta = {}

        # fill meta with defaults if not already present
        w, h = image.size
        meta_from_image = {
            'offset': np.array((0.0, 0.0)),
            'scale': np.array((1.0, 1.0)),
            'rotation': {'angle': 0.0, 'width': None, 'height': None},
            'valid_area': np.array((0.0, 0.0, w - 1, h - 1)),
            'hflip': False,
            'width_height': np.array((w, h)),
        }
        for k, v in meta_from_image.items():
            if k not in meta:
                meta[k] = v

        return image, anns, meta


class RescaleAbsolute(openpifpaf.transforms.Preprocess):
    def __init__(self, long_edge, *, fast=False, resample=PIL.Image.BICUBIC):
        self.long_edge = long_edge
        self.fast = fast
        self.resample = resample


    def __call__(self, image, anns, meta):
        w, h = image.size
        this_long_edge = self.long_edge
        if isinstance(this_long_edge, (tuple, list)):
            this_long_edge = torch.randint(
                int(this_long_edge[0]),
                int(this_long_edge[1]), (1,)
            ).item()

        s = this_long_edge / max(h, w)
        if h > w:
            target_w, target_h = int(w * s), int(this_long_edge)
        else:
            target_w, target_h = int(this_long_edge), int(h * s)
        return _scale(image, anns, meta, target_w, target_h,
                      self.resample, fast=self.fast)


class CropTopOut(openpifpaf.transforms.Preprocess):
    def __init__(self, top_ratio, height_stride=None):
        self.top_ratio = top_ratio
        self.height_stride = height_stride


    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)
        original_valid_area = meta['valid_area'].copy()

        w, h = image.size
        y_offset = int(h * self.top_ratio)
        if self.height_stride is not None:
            new_h = h - y_offset
            new_h = self.height_stride * round((new_h-1)/self.height_stride) + 1
            y_offset = h - new_h
        LOG.debug('top crop offset %d', y_offset)
        ltrb = (0, y_offset, w, h)
        image = image.crop(ltrb)

        # Shift annotations
        for ann in anns:
            if ann['object_type'] is JaadType.PEDESTRIAN:
                ann['box'][1] -= y_offset
                ann['center'][1] -= y_offset

        ltrb = np.array(ltrb)
        meta['offset'] += ltrb[:2]

        new_wh = image.size
        LOG.debug('valid area before crop of %s: %s', ltrb, original_valid_area)
        # Process crops from left and top
        meta['valid_area'][:2] = np.maximum(0.0, original_valid_area[:2] - ltrb[:2])
        # Process crops from right and bottom
        new_rb_corner = original_valid_area[:2] + original_valid_area[2:] - ltrb[:2]
        new_rb_corner = np.maximum(0.0, new_rb_corner)
        new_rb_corner = np.minimum(new_wh, new_rb_corner)
        meta['valid_area'][2:] = new_rb_corner - meta['valid_area'][:2]
        LOG.debug('valid area after crop: %s', meta['valid_area'])

        return image, anns, meta


class ZoomInOrOut(openpifpaf.transforms.Preprocess):
    def __init__(self, scale_range=(0.95, 1.05), *, fast=False,
                 resample=PIL.Image.BICUBIC):
        self.scale_range = scale_range
        self.fast = fast
        self.resample = resample


    def __call__(self, image, anns, meta):
        w, h = image.size
        scale_factor = (
            self.scale_range[0] +
            torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0])
        )
        new_w, new_h = round(w * scale_factor), round(h * scale_factor)
        image, anns, meta = _scale(image, anns, meta, new_w, new_h,
                                   self.resample, fast=self.fast)

        if scale_factor < 1.0: # pad image to original size
            x_offset = int(torch.randint(0, w - new_w + 1, (1,)).item())
            y_offset = int(torch.randint(0, h - new_h + 1, (1,)).item())
            ltrb = (x_offset, y_offset, w - new_w - x_offset, h - new_h - y_offset)
            image = torchvision.transforms.functional.pad(
                image, ltrb, fill=(124, 116, 104))

            # Shift annotations
            for ann in anns:
                if ann['object_type'] is JaadType.PEDESTRIAN:
                    ann['box'][0] += x_offset
                    ann['box'][1] += y_offset
                    ann['center'][0] += x_offset
                    ann['center'][1] += y_offset
            ltrb = np.array(ltrb)
            meta['offset'] -= ltrb[:2]
            LOG.debug('valid area before pad with %s: %s', ltrb, meta['valid_area'])
            meta['valid_area'][:2] += ltrb[:2]
            LOG.debug('valid area after pad: %s', meta['valid_area'])

        elif scale_factor > 1.0: # crop image to original size
            x_offset = int(torch.randint(0, new_w - w + 1, (1,)).item())
            y_offset = int(torch.randint(0, new_h - h + 1, (1,)).item())
            ltrb = (x_offset, y_offset, x_offset + w, y_offset + h)
            image = image.crop(ltrb)

            # Shift and crop annotations
            for ann in anns:
                if ann['object_type'] is JaadType.PEDESTRIAN:
                    ann['box'][0] -= x_offset
                    ann['box'][1] -= y_offset
                    ann['center'][0] -= x_offset
                    ann['center'][1] -= y_offset
                    if ann['box'][0] < 0:
                        max_x = ann['box'][0] + ann['box'][2]
                        ann['box'][0] = 0
                        ann['box'][2] = max_x
                        ann['center'][0] = .5*max_x
                        ann['width'] = max_x
                    if ann['box'][1] < 0:
                        max_y = ann['box'][1] + ann['box'][3]
                        ann['box'][1] = 0
                        ann['box'][3] = max_y
                        ann['center'][1] = .5*max_y
                        ann['height'] = max_y
                    if ann['box'][0] + ann['box'][2] > w - 1:
                        new_width = w - 1 - ann['box'][0]
                        ann['box'][2] = new_width
                        ann['center'][0] = ann['box'][0] + .5*new_width
                        ann['width'] = new_width
                    if ann['box'][1] + ann['box'][3] > h - 1:
                        new_height = h - 1 - ann['box'][1]
                        ann['box'][3] = new_height
                        ann['center'][1] = ann['box'][1] + .5*new_height
                        ann['height'] = new_height
            # Remove annotation if out of bound
            anns = [ann for ann in anns if not (
                        ann['object_type'] is JaadType.PEDESTRIAN
                        and ((ann['width'] < 5) or (ann['height'] < 5))
                    )]

            ltrb = np.array(ltrb)
            meta['offset'] += ltrb[:2]
            new_wh = image.size
            original_valid_area = meta['valid_area'].copy()
            LOG.debug('valid area before crop of %s: %s', ltrb, original_valid_area)
            # Process crops from left and top
            meta['valid_area'][:2] = np.maximum(0.0, original_valid_area[:2] - ltrb[:2])
            # Process crops from right and bottom
            new_rb_corner = original_valid_area[:2] + original_valid_area[2:] - ltrb[:2]
            new_rb_corner = np.maximum(0.0, new_rb_corner)
            new_rb_corner = np.minimum(new_wh, new_rb_corner)
            meta['valid_area'][2:] = new_rb_corner - meta['valid_area'][:2]
            LOG.debug('valid area after crop: %s', meta['valid_area'])

        return image, anns, meta


class HFlip(openpifpaf.transforms.Preprocess):
    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        w, _ = image.size
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        for ann in anns:
            if ann['object_type'] is JaadType.PEDESTRIAN:
                ann['box'][0] = -(ann['box'][0] + ann['box'][2]) - 1.0 + w
                ann['center'][0] = -ann['center'][0] - 1.0 + w
                ann['bag_left_side'], ann['bag_right_side'] = (
                    ann['bag_right_side'], ann['bag_left_side'])
                ann['pose_left'], ann['pose_right'] = (
                    ann['pose_right'], ann['pose_left'])

        assert meta['hflip'] is False
        meta['hflip'] = True
        meta['valid_area'][0] = -(meta['valid_area'][0] + meta['valid_area'][2]) + w

        return image, anns, meta


class ToAnnotations(openpifpaf.transforms.Preprocess):
    def __init__(self, object_annotations):
        self.object_annotations = object_annotations


    def __call__(self, image, anns, meta):
        anns = [
            self.object_annotations[ann['object_type']](**ann)
            for ann in anns
        ]
        return image, anns, meta


def replaceNormalization(compose_transform):
    new_preprocess_list = []
    for op in compose_transform.preprocess_list:
        if isinstance(op, openpifpaf.transforms.NormalizeAnnotations):
            new_preprocess_list.append(NormalizeAnnotations())
        elif isinstance(op, openpifpaf.transforms.Compose):
            new_preprocess_list.append(replaceNormalization(op))
        else:
            new_preprocess_list.append(op)
    return openpifpaf.transforms.Compose(new_preprocess_list)


TRAIN_TRANSFORM = replaceNormalization(openpifpaf.transforms.TRAIN_TRANSFORM)
EVAL_TRANSFORM = replaceNormalization(openpifpaf.transforms.EVAL_TRANSFORM)
