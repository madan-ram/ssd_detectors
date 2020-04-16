# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from preprocessing import ssd_vgg_preprocessing


def get_preprocessing(name, is_training=False):
    """Returns preprocessing_fn(image, height, width, **kwargs).

    Args:
        name: The name of the preprocessing function.
        is_training: `True` if the model is being used for training.

    Returns:
        preprocessing_fn: A function that preprocessing a single image (pre-batch).
            It has the following signature:
                image = preprocessing_fn(image, output_height, output_width, ...).

    Raises:
        ValueError: If Preprocessing `name` is not recognized.
    """

    preprocessing_fn_map = {
        'TBPP512_dense_separable': ssd_vgg_preprocessing,
        'DSODTBPP512': ssd_vgg_preprocessing,
        'TBPP512': ssd_vgg_preprocessing,
        'SSD512_resnet': ssd_vgg_preprocessing
    }

    if name not in preprocessing_fn_map:
        raise ValueError('------->Preprocessing name [%s] was not recognized<-------' % name)

    def preprocessing_fn(out_shape, is_training=is_training, resize=preprocessing_fn_map[name].Resize.PAD_AND_RESIZE, **kwargs):
        def _preprocessing_fn(tfrecord):

            image = tf.io.decode_image(tfrecord['image/encoded'], channels=3)
            image = tf.reshape(image, (tfrecord['image/height'][0], tfrecord['image/width'][0], tf.constant([3], dtype=tf.int64)[0]))
            boxes, labels = tf.sparse.to_dense(tfrecord['boxes']), tf.sparse.to_dense(tfrecord['labels'])

            image, labels, bboxes = preprocessing_fn_map[name].preprocess_image(image, labels, boxes, out_shape, is_training=is_training, resize=resize, **kwargs)

            tfrecord['input'] = image
            tfrecord['boxes'] = boxes
            tfrecord['labels'] = labels
            tfrecord['image_shape'] = out_shape

            return tfrecord
        return _preprocessing_fn

    return preprocessing_fn


