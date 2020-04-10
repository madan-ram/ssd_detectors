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
"""Contains a factory for building various models.
"""

import functools
import tensorflow as tf

from models import tbpp_model


model_fn_map = {
    'TBPP512_dense_separable': tbpp_model.TBPP512_dense_separable,
    'DSODTBPP512': tbpp_model.DSODTBPP512,
    'TBPP512': tbpp_model.TBPP512
}


# arg_scopes_map = {
#                 'vgg_a': vgg.vgg_arg_scope,
#                 'vgg_16': vgg.vgg_arg_scope,
#                 'vgg_19': vgg.vgg_arg_scope,
                
#                 'ssd_avt_vgg': ssd_avt_vgg.ssd_arg_scope,
#                 'ssd_avt_vgg_caffe': ssd_avt_vgg.ssd_arg_scope_caffe,
                
#                 'ssd_avt_vgg_deep': ssd_avt_vgg_deep.ssd_arg_scope,
#                 'ssd_avt_vgg_deep_caffe': ssd_avt_vgg_deep.ssd_arg_scope_caffe,
                
#                 'ssd_resnet_v2': ssd_resnet_v2.ssd_arg_scope,
#                 'ssd_resnet_v2_caffe': ssd_resnet_v2.ssd_arg_scope_caffe,
#               }

# networks_obj = {
#                 'ssd_avt_vgg_deep': ssd_avt_vgg_deep.SSDNet,
#                 'ssd_resnet_v2': ssd_resnet_v2.SSDNet,
#               }


# def get_network(name):
#     """Get a network object from a name.
#     """
#     # params = networks_obj[name].default_params if params is None else params
#     return networks_obj[name]


def get_model(name, num_classes, input_shape=(512, 512, 3), softmax=True):
    """Returns a KerasModel such as `logits, end_points = (images)`.

    Args:
      name: The name of the network.
      num_classes: Number of classes
      input_shape: INput shape
      softmax: `True` if the softmax else sigmoid
    Returns:
      model: Keras model

    Raises:
      ValueError: If network `name` is not recognized.
    """
    if name not in model_fn_map:
        raise ValueError('Name of network unknown %s' % name)

    model_fn = model_fn_map[name]

    return model_fn(num_classes, input_shape=(512, 512, 3), softmax=True)
