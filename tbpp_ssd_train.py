# Copyright 2016 Paul Balanca. All Rights Reserved.
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
"""Generic training script that trains a SSD model using a given dataset."""
import tensorflow as tf
from tensorflow.keras.models import Model
from models import models_factory
import os

from tbpp_utils import PriorUtil
from ssd_data import InputGenerator
from tbpp_training import TBPPFocalLoss
from data_synthtext import GTUtility

# =========================================================================== #
# SSD Network flags.
# =========================================================================== #
tf.compat.v1.app.flags.DEFINE_float(
    'loss_alpha', 1., 'Alpha parameter in the loss function.')
tf.compat.v1.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.compat.v1.app.flags.DEFINE_float(
    'match_threshold', 0.5, 'Matching threshold in the loss function.')

# =========================================================================== #
# General Flags.
# =========================================================================== #
tf.compat.v1.app.flags.DEFINE_string(
    'model_log_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.compat.v1.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
# =========================================================================== #
# Dataset Flags.
# =========================================================================== #

tf.compat.v1.app.flags.DEFINE_float(
    'dataset_split_percentage', 0.8, 'Train dataset percentage')

tf.compat.v1.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
# =========================================================================== #
tf.compat.v1.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')
tf.compat.v1.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')
tf.compat.v1.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
tf.compat.v1.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')
tf.compat.v1.app.flags.DEFINE_string(
    'model_name', 'TBPP512', 'The name of the architecture to train.')
# =========================================================================== #
tf.compat.v1.app.flags.DEFINE_integer(
    'image_size', None, 'Image size')
# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.compat.v1.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.compat.v1.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.compat.v1.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.compat.v1.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.compat.v1.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')



# =========================================================================== #
tf.compat.v1.app.flags.DEFINE_float(
    'use_tpu', -1, 
    'Use TPU compution')

FLAGS = tf.compat.v1.app.flags.FLAGS

# =========================================================================== #
# Main training routine.
# =========================================================================== #

def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

def main(_):

    # TODO: remove below freeze
    freeze = []

    epochs = 100

    # Load config and paramaters
    softmax = True
    input_shape = (FLAGS.image_size, FLAGS.image_size, 3)
    TPU_ADDRESS = None
    try:
        device_name = os.environ['COLAB_TPU_ADDR']
        TPU_ADDRESS = 'grpc://' + device_name
        print('Found TPU at: {}'.format(TPU_ADDRESS))
        FLAGS.use_tpu = 1
    except KeyError:
        FLAGS.use_tpu = -1
        print('TPU not found')

    # Initialize traininglr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)
    tf.keras.backend.clear_session()
    tf.config.set_soft_device_placement(1)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

    logdir = os.path.join(model_log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    strategy = None
    if FLAGS.use_tpu == 1:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_ADDRESS)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)

    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

    with strategy.scope(), tf.Graph().as_default():
        
        # Get the SSD network and its anchors.
        model = models_factory.get_model(FLAGS.model_name, FLAGS.num_classes, input_shape=input_shape, softmax=softmax)
        # Get model summary
        model.summary()

        # Get data
        gt_util = GTUtility('data/SynthText/', polygon=True)

        prior_util = PriorUtil(model)
        gt_util_train, gt_util_val = gt_util.split(FLAGS.dataset_split_percentage)

        gen_train = InputGenerator(gt_util_train, prior_util, FLAGS.batch_size, model.image_size)
        gen_val = InputGenerator(gt_util_val, prior_util, FLAGS.batch_size, model.image_size)

        # # Create checkpoint dir if does not exists
        # checkdir = os.path.join(FLAGS.model_log_dir, 'checkpoints/')
        # if not os.path.exists(checkdir):
        #     os.makedirs(checkdir)

        lr_callback = keras.callbacks.LearningRateScheduler(step_decay)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)

        loss = TBPPFocalLoss(lambda_conf=10000.0, lambda_offsets=1.0)

        model.compile(optimizer=optimizer, loss=loss.compute, metrics=loss.metrics)


        model.fit_generator(
            gen_train.generate(), 
            # batch_size=FLAGS.batch_size,
            epochs=epochs,
            verbose=1, 
            # callbacks=None,
            shuffle=True,
            # class_weight=None,
            # sample_weight=None, 
            initial_epoch=0, 
            validation_data=gen_val.generate(), 
            validation_steps=gen_val.num_batches, 
            steps_per_epoch=gen_train.num_batches,
            # validation_steps=None, 
            # validation_freq=1,
            max_queue_size=16, 
            workers=8,
            use_multiprocessing=True,
            callbacks=[tensorboard_callback, lr_callback],
        )


if __name__ == '__main__':
    tf.compat.v1.app.run()
