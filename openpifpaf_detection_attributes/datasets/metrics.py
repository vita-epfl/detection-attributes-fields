import logging
import math
from typing import List

import openpifpaf

from .headmeta import AttributeMeta


LOG = logging.getLogger(__name__)


def compute_iou(pred_c, pred_w, pred_h, gt_box):
    inter_box = [
        max(pred_c[0] - .5*pred_w, gt_box[0]),
        max(pred_c[1] - .5*pred_h, gt_box[1]),
        min(pred_c[0] + .5*pred_w, gt_box[0] + gt_box[2]),
        min(pred_c[1] + .5*pred_h, gt_box[1] + gt_box[3])
    ]
    inter_area = (
        max(0., inter_box[2] - inter_box[0])
        * max(0., inter_box[3] - inter_box[1])
    )
    pred_area = pred_w * pred_h
    gt_area = gt_box[2] * gt_box[3]
    iou = (
        inter_area / (pred_area + gt_area - inter_area)
        if pred_area + gt_area - inter_area != 0 else 0.
    )
    return iou


def compute_ap(self, stats):
    tps = [tp for _, tp in sorted(zip(stats['score'],
                                      stats['tp']),
                                  key=lambda pair: pair[0],
                                  reverse=True)]
    fps = [fp for _, fp in sorted(zip(stats['score'],
                                      stats['fp']),
                                  key=lambda pair: pair[0],
                                  reverse=True)]
    cumsum = 0
    for idx, val in enumerate(tps):
        tps[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(fps):
        fps[idx] += cumsum
        cumsum += val
    recs = tps[:]
    for idx, val in enumerate(tps):
        recs[idx] = (
            float(tps[idx]) / stats['n_gt']
            if stats['n_gt'] != 0 else 0.
        )
    precs = tps[:]
    for idx, val in enumerate(tps):
        precs[idx] = (
            float(tps[idx]) / (tps[idx] + fps[idx])
            if tps[idx] + fps[idx] != 0 else 0.
        )
    return average_precision(recs, precs)


def average_precision(rec, prec):
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap


class InstanceDetection(openpifpaf.metric.base.Base):
    def __init__(self, attribute_metas: List[AttributeMeta]):
        self.attribute_metas = attribute_metas
        assert len(self.attribute_metas) > 0
        self.attributes = [att for att in self.attribute_metas
                           if ((att.attribute == 'confidence')
                               or (att.group != 'detection'))]

        self.det_stats = {}
        for attribute in self.attributes:
            if attribute.is_classification:
                n_classes = max(attribute.n_channels, 2)
            else:
                n_classes = 10
            self.det_stats[attribute['attribute']] = {'n_classes': n_classes}
            for cls in range(n_classes):
                self.det_stats[attribute['attribute']][cls] = {
                    'n_gt': 0, 'score': [], 'tp': [], 'fp': []}


    def accumulate(self, predictions, image_meta, *, ground_truth=None):
        for attribute in self.attributes:
            self.accumulate_attribute(attribute, predictions, image_meta,
                                      ground_truth=ground_truth)


    def accumulate_attribute(self, attribute, predictions, image_meta, *,
                             ground_truth=None):
        for cls in range(self.det_stats[attribute['attribute']]['n_classes']):
            det_stats = self.det_stats[attribute['attribute']][cls]

            # Initialize ground truths
            gt_match = {}
            for gt in ground_truth:
                if (
                    gt['ignore_eval']
                    or gt[attribute['attribute']] is None
                    or (not attribute['is_classification'])
                    or int(gt[attribute['attribute']]) == cls
                ):
                    gt_match[gt['id']] = False
                    if (
                        (not gt['ignore_eval'])
                        and (gt[attribute['attribute']] is not None)
                    ):
                        det_stats['n_gt'] += 1

            # Rank predictions based on confidences
            ranked_preds = []
            for pred in prediction:
                if attribute['attribute'] in pred:
                    rpred = copy.deepcopy(pred)
                    pred_score = pred[attribute['attribute']]
                    if (
                        (attribute['attribute'] == 'confidence')
                        or (not attribute['is_classification'])
                    ):
                        rpred['score'] = pred['confidence']
                    elif (
                        attribute['is_classification']
                        and (attribute['n_channels'] == 1)
                    ):
                        rpred['score'] = (
                            (cls*pred_score + (1-cls)*(1.-pred_score))
                            * pred['confidence']
                        )
                    elif (
                        attribute['is_classification']
                        and (attribute['n_channels'] > 1)
                    ):
                        rpred['score'] = pred_score[cls] * pred['confidence']
                    ranked_preds.append(rpred)
            ranked_preds.sort(key=lambda x:x['score'], reverse=True)

            # Match predictions with closest groud truths
            for pred in ranked_preds:
                max_iou = -1.
                match = None
                for gt in gt_match:
                    if ('width' in pred) and ('height' in pred):
                        iou = compute_iou(pred['center'], pred['width'],
                                          pred['height'], gt['box'])
                    else:
                        iou = 0.
                    if (iou > 0.5) and (iou >= max_iou):
                        if (
                            (gt[attribute['attribute']] is None)
                            or attribute.is_classification
                            or (abs(gt[attribute['attribute']]
                                -pred[attribute['attribute']]) <= (cls+1)*.5)
                        ):
                            max_iou = iou
                            match = gt

                # Classify predictions as True Positives or False Positives
                if match is not None:
                    if ((not match['ignore_eval'])
                        and (match[attribute['attribute']] is not None)):
                        if not gt_match['box'][match]:
                            # True positive
                            det_stats['score'].append(pred['score'])
                            det_stats['tp'].append(1)
                            det_stats['fp'].append(0)

                            gt_match[match['id']] = True
                        else:
                            # False positive (multiple detections)
                            det_stats['score'].append(pred['score'])
                            det_stats['tp'].append(0)
                            det_stats['fp'].append(1)
                    else:
                        # Ignore instance
                        pass
                else:
                    # False positive
                    det_stats['score'].append(pred['score'])
                    det_stats['tp'].append(0)
                    det_stats['fp'].append(1)


    def stats(self):
        text_labels = []
        stats = []

        att_aps = []
        for attribute in self.attributes:
            cls_aps = []
            for cls in range(self.det_stats[attribute['attribute']]['n_classes']):
                cls_ap = compute_ap(self.det_stats[attribute['attribute']][cls])
                cls_aps.append(cls_ap)
            if attribute['attribute'] == 'confidence':
                text_labels.append('detection_AP')
                stats.append(cls_aps[1])
                att_aps.append(cls_aps[1])
                LOG.info('detection AP = {}'.format(cls_aps[1]))
            else:
                text_labels.append(attribute['attribute'] + '_AP')
                att_ap = sum(cls_aps) / len(cls_aps)
                stats.append(att_ap)
                att_aps.append(att_ap)
                LOG.info('{} AP = {}'.format(attribute['attribute'], att_ap))
        text_labels.append('attribute_mAP')
        map = sum(att_aps) / len(att_aps)
        stats.append(map)
        LOG.info('attribute mAP = {}'.format(map))

        data = {
            'text_labels': self.text_labels,
            'stats': stats,
        }
        return data


    def write_predictions(self, filename, *, additional_data=None):
        raise NotImplementedError
