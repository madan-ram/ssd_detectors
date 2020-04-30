import os, sys
sys.path.append(os. getcwd())

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from datasets import ssd_utils
from models import models_factory
from enum import Enum, IntEnum

import matplotlib.pyplot as plt
import numpy as np
import cv2

AUTO = tf.data.experimental.AUTOTUNE

def tfrecord_to_tensor(data_single_record):
    keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.io.FixedLenFeature([1], tf.int64),
        'image/width': tf.io.FixedLenFeature([1], tf.int64),
        'image/channels': tf.io.FixedLenFeature([1], tf.int64),
        'image/shape': tf.io.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.io.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.io.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.io.VarLenFeature(dtype=tf.int64),
    }
    # decode the TFRecord
    tfrecord = tf.io.parse_single_example(data_single_record, keys_to_features)

    xmin = tfrecord['image/object/bbox/xmin']
    ymin = tfrecord['image/object/bbox/ymin']
    xmax = tfrecord['image/object/bbox/xmax']
    ymax = tfrecord['image/object/bbox/ymax']
    labels = tfrecord['image/object/bbox/label']

    boxes = tf.sparse.concat(1, [tf.sparse.reshape(x, (-1, 1)) for x in [ymin, xmin, ymax, xmax]])
    tfrecord['boxes'] = boxes
    tfrecord['labels'] = labels

    # Delete unwanted data after encoding
    del tfrecord['image/object/bbox/xmin']
    del tfrecord['image/object/bbox/ymin']
    del tfrecord['image/object/bbox/xmax']
    del tfrecord['image/object/bbox/ymax']
    del tfrecord['image/object/bbox/label']

    return tfrecord


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
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
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

def encode_label(ssd_anchors, num_classes):

    def _encode_label(tfrecord):
        boxes = tfrecord['boxes']
        labels = tfrecord['labels']

        g_target_labels, g_target_boxes, g_target_scores = ssd_utils.bboxes_encode(labels, boxes, ssd_anchors, num_classes)

        print(g_target_boxes)
        # TODO: Need to check this
        g_target_boxes = [tf.reshape(bb, [-1, 4]) for bb in g_target_boxes]

        g_target_scores = tf.cast(tf.concat([tf.reshape(score, [-1, 1]) for score in g_target_scores], axis=0), dtype='float32')
        tfrecord['scores'] = g_target_scores

        g_target_labels = tf.concat([tf.reshape(label, [-1]) for label in g_target_labels], axis=0)
        onhot_label = tf.eye(num_classes, dtype="float32")
        g_target_labels = tf.gather(onhot_label, g_target_labels)

        tfrecord['labels'] = g_target_labels

        g_target_boxes = tf.concat(values=g_target_boxes, axis=0)
        tfrecord['boxes'] = g_target_boxes

        print(g_target_boxes)
        print(g_target_labels)
        tfrecord['targets'] = tf.concat(values=[g_target_boxes, g_target_labels], axis=-1)

        print(tfrecord['targets'])
        return tfrecord

    return _encode_label

def read_image(tfrecord):
    image = tf.io.decode_image(tfrecord['image/encoded'])
    del tfrecord['image/encoded']
    tfrecord['input'] = image
    return tfrecord

def get_dataset(tfrecord_path_list, ssd_anchors, num_classes, batch_size=32, preprocess_fn=None, preprocess_fn_args={}):
    """Gets a dataset tuple with instructions for reading Pascal VOC dataset.
    Args:
      split_name: A train/test split name.
      dataset_dir: The base directory of the dataset sources.
      file_pattern: The file pattern to use when matching the dataset sources.
        It is assumed that the pattern contains a '%s' string so that the split
        name can be inserted.
      reader: The TensorFlow reader type.

    Returns:
      A `Dataset` namedtuple.

    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    # Create dataset object
    # Read and featurize 
    dataset = tf.data.TFRecordDataset(tfrecord_path_list, num_parallel_reads=AUTO)

    dataset = dataset.map(tfrecord_to_tensor, num_parallel_calls=AUTO)

    # Set the number of datapoints you want to load and shuffle
    seed = int(time.mktime(datetime.now().timetuple()))
    dataset = dataset.shuffle(buffer_size=5000, seed=seed)

    # Preprocess dataset
    if preprocess_fn is not None:
        dataset = dataset.map(preprocess_fn(**preprocess_fn_args), num_parallel_calls=AUTO)
    else:
        # if Process is non assum that the images are already resized hence load as it is, if the size mismatch then dataset will throw error
        print('------->Preprocessing is disable since input to preprocess_fn is None')
        dataset = dataset.map(read_image, num_parallel_calls=AUTO)

    # # Prepare data
    dataset = dataset.map(encode_label(ssd_anchors, num_classes), num_parallel_calls=AUTO)


    # # This dataset will go on forever
    # dataset = dataset.repeat()

    # dataset = dataset.batch(batch_size)

    # dataset = dataset.prefetch(buffer_size=AUTO)

    return dataset

if __name__ == '__main__':
    from preprocessing.preprocessing_factory import get_preprocessing

    softmax = True
    input_shape=(512, 512, 3)
    num_classes = 21
    model_name = 'SSD512_resnet'
    model = models_factory.get_model(model_name, num_classes, input_shape=input_shape, softmax=softmax)
    list_of_tfrecord_path = tf.io.gfile.glob("/Users/madanram/SoulOfCoder/SSD-Tensorflow-V2/datasets/VOC2012/tf/avt_2020_v1_*.tfrecord")
    
    feat_shapes = [layer.get_shape().as_list() for layer in model.source_layers]
    anchor_ratios = model.aspect_ratios
    anchor_sizes = model.minmax_sizes
    anchor_steps = model.steps
    special_ssd_boxes = model.special_ssd_boxes
    anchor_offset = 0.5

    ssd_anchors = ssd_utils.anchors(
        input_shape, 
        feat_shapes,
        anchor_sizes,
        anchor_ratios,
        anchor_steps, 
        anchor_offset
    )
    preprocess_fn = get_preprocessing('SSD512_resnet', is_training=True)
    print(preprocess_fn, '--------------------------')
    dataset = get_dataset(list_of_tfrecord_path, ssd_anchors, num_classes, preprocess_fn=preprocess_fn, preprocess_fn_args={'out_shape': input_shape})

    print('Iterating over data-sample ........................')

    for tfrecord in dataset:
        x = tfrecord['input']
        print(tfrecord['boxes'])
        break
        # print(y.shape, '#!#########@!#!#!#!#!##!!#')
        # print(x.shape, '$$$$$@$!#!#!#!$!!$$!$!!')

        # idx = np.random.randint(32, size=1)
        # vis_img = np.asarray(x[idx[0]], dtype=np.uint8)
        # print(vis_img.dtype)
        # y = np.asarray(y, dtype=np.int32)

        # for prior in y[idx[0]]:
        #     ymin, xmin, ymax, xmax  = prior[:4]
        #     classes = prior[4:]
        #     if classes[0] != 1.0:
        #         # CLass zero is background
        #         vis_img = cv2.rectangle(vis_img, (ymin, xmin), (ymax, xmax), (255, 0, 0), thickness=1)
        # plt.imshow(vis_img)
        # print(np.sum(x))
        # plt.show()
        # break

    # for data in dataset:
    #     print(data['input'].shape)
    #     print(data['targets'].numpy().shape)
    #     print(data['labels'].shape)
    #     break
        