from datasets import ssd_common
import tensorflow as tf
import numpy as np
import math

def anchors(img_shape, feat_shapes, anchor_sizes, anchor_ratios, anchor_steps, anchor_offset, dtype=np.float32):
    """Compute the default anchor boxes, given an image shape.
    """
    return ssd_anchors_all_layers(img_shape,
                                  feat_shapes,
                                  anchor_sizes,
                                  anchor_ratios,
                                  anchor_steps,
                                  anchor_offset,
                                  dtype)

def bboxes_encode(labels, bboxes, anchors, num_classes, scope=None):
    """Encode labels and bounding boxes.
    """
    target_labels, target_boxes, iou_scores = ssd_common.tf_ssd_bboxes_encode(
        labels, bboxes, anchors,
        num_classes,
        prior_scaling=[0.1, 0.1, 0.2, 0.2],
        ignore_threshold=0.5)

    return target_labels, target_boxes, iou_scores


def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    print(feat_shape, '-------------------------------')
    y, x = np.mgrid[0:feat_shape[1], 0:feat_shape[2]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes) + len(ratios)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    # Add first anchor boxes with ratio=1.
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i+di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i+di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w

def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors

def bboxes_decode(feat_localizations, anchors,
                  scope='ssd_bboxes_decode'):
    """Encode labels and bounding boxes.
    """
    return ssd_common.tf_ssd_bboxes_decode(
        feat_localizations, anchors,
        prior_scaling=[0.1, 0.1, 0.2, 0.2],
        scope=scope)