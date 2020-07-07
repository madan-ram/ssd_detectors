# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datasets import avt_dataset

import tensorflow as tf
import glob


datasets_map = {
    'avt_dataset': avt_dataset,
    'avt_2020_v1': avt_dataset
}

def get_dataset(name, tfrecord_path, ssd_anchors, num_classes, preprocess_fn=None, preprocess_fn_args={}, batch_size=32):
    """Given a dataset name and a  returns a Dataset.

    Args:
        name: String, the name of the dataset.
        tfrecord_path: The file path to tensor-record
        file_pattern: The file pattern to use for matching the dataset source files.
        reader: The subclass of tf.ReaderBase. If left as `None`, then the default
            reader defined by each dataset is used.
    Returns:
        A `Dataset` class.
    Raises:
        ValueError: If the dataset `name` is unknown.
    """
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    return datasets_map[name].get_dataset(tf.io.gfile.glob(tfrecord_path), ssd_anchors, num_classes, preprocess_fn=preprocess_fn, preprocess_fn_args=preprocess_fn_args, batch_size=batch_size)

if __name__ == '__main__':
    tfrecord_path = "/Users/madanram/SoulOfCoder/SSD-Tensorflow-V2/datasets/VOC2012/tf/avt_2020_v1_*.tfrecord"
    get_dataset('avt_dataset', tfrecord_path)
    

   
