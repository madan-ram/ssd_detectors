# Copyright 2015 Paul Balanca. All Rights Reserved.
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
"""Converts Pascal VOC data to TFRecords file format with Example protos.

The raw Pascal VOC data set is expected to reside in JPEG files located in the
directory 'JPEGImages'. Similarly, bounding box annotations are supposed to be
stored in the 'Annotation directory'

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFREcord
file contains ~1000 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'JPEG'


    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import os
import sys
import random
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
from pascalvoc_to_tfrecords import Parser

# # TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200

def negative_to_zero(val):
    if val<0.0:
        return 0.0
    return val

def above_1_to_1(val):
    if val>1.0:
        return 1.0
    return val

def _convert_to_example(image_data, labels, labels_text, bboxes, shape,
                        difficult, truncated):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(xml_file_list, output_dir, name='voc_train', shuffling=False):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    label_id_map = {label: idx for idx, label in enumerate(config.whitelist_with_fields_label)}

    fidx = 0
    tf_filename = _get_output_filename(output_dir, name, fidx)
    tfrecord_writer = tf.io.TFRecordWriter(tf_filename)
    count_num_exp = 0
    for idx, xml_file_path in enumerate(xml_file_list):
        sys.stdout.write('\r>> Converting image %d/%d' % (idx+1, len(xml_file_list)))
        sys.stdout.flush()

        if (idx+1)%SAMPLES_PER_FILES == 0:
            # Close previous writer and open new writer
            tfrecord_writer.close()
            fidx += 1

            tf_filename = _get_output_filename(output_dir, name, fidx)
            tfrecord_writer = tf.io.TFRecordWriter(tf_filename)

        parser_genr = Parser(xml_file_path, ignore_label_lst=['unlabelled'])
        image_path = parser_genr.img_path
        image_width = parser_genr.img_width
        image_height = parser_genr.img_height
        
        image_data = tf.compat.v1.gfile.FastGFile(image_path, 'rb').read()

        assert image_data.shape[0] != image_height or image_data.shape[1] != image_width, "Image and xml size mismatch for image path->{}".format(image_path)

        shape = [
            image_height,
            image_width,
            image_data.shape[2]
        ]

        difficult = []
        truncated = []
        labels = []
        bboxes = []
        labels_text = []
        for xml_info in parser_genr.generator():
            label_name, xmin, ymin, xmax, ymax, text, _, _ = xml_info

            label_id = label_id_map.get(label_name, None)

            if label_id == None:
                continue

            labels.append(label_id)
            labels_text.append(label_name)
            bboxes.append((ymin, xmin, ymax, xmax))


        example = _convert_to_example(image_data, labels, labels_text, bboxes, shape, difficult, truncated)
        tfrecord_writer.write(example.SerializeToString())
        count_num_exp += 1

    print("\nNumer of samples %d"%count_num_exp)