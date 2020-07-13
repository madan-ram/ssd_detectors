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
import config
import math
from ssd_trainer import SSDFocalLoss,AvtMetricsCallback
# , compute_metrics
from datetime import datetime
from tensorflow.python.framework.ops import disable_eager_execution
from preprocessing.preprocessing_factory import get_preprocessing
from datasets.dataset_factory import get_dataset
from datasets import ssd_utils
from tensorflow.keras.models import load_model
import tensorflow.python.keras.backend as K
import pickle
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2

# Configure
tf.keras.backend.clear_session()
tf.config.set_soft_device_placement(1)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

#print(tf.executing_eagerly())
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

tf.compat.v1.app.flags.DEFINE_integer(
    'load_model_epoch', 0,
    'If set to any number > 0 will load from model_log_dir')

tf.compat.v1.app.flags.DEFINE_string(
    'dataset_dir', '/tmp/tf/',
    'Directory where tfrecord placed.')

tf.compat.v1.app.flags.DEFINE_string(
    'dataset_name', 'avt_2020_v1',
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
tf.compat.v1.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
tf.compat.v1.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')
tf.compat.v1.app.flags.DEFINE_string(
    'model_name', 'SSD512_resnet', 'The name of the architecture to train.')
tf.compat.v1.app.flags.DEFINE_string(
    'preprocess_name', 'SSD512_resnet', 'The name of the processing pipeline to train.')
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


def set_gpu_mem_config():
        """Allocate only as much GPU memory as needed for Keras.
        Based on runtime allocations: it starts out allocating very little memory,
        and as Sessions get run and more GPU memory is needed, we extend the GPU
        memory region needed by the TensorFlow process. Note that we do not release
        memory, since that can lead to even worse memory fragmentation.
        """
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.70)
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        K.set_session(tf.compat.v1.Session(config=config))


def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

def get_train_iter(tfrecord):
    return (tfrecord['input'], tfrecord['targets'])

def main(_):

    # TODO: remove below freeze
    freeze = []
    steps_per_epoch=1
    val_steps_per_epoch=1

    model_dir = os.path.join(FLAGS.model_log_dir, 'models')
    model_params_path = os.path.join(model_dir, 'ssd_params.json')
    model_str = 'model.{epoch:04d}.tf'
    model_path = os.path.join(model_dir, model_str)


    epochs = 1000
    set_gpu_mem_config()

    # Load config and paramaters
    softmax = True
    input_shape = (FLAGS.image_size, FLAGS.image_size, 3)
    # Get data
    train_ref_path = os.path.join(FLAGS.dataset_dir+'/training_data', FLAGS.dataset_name+'*.tfrecord')
    val_ref_path = os.path.join(FLAGS.dataset_dir+'/validation_data', FLAGS.dataset_name+'*.tfrecord')

    TPU_ADDRESS = None
    try:
        device_name = os.environ['COLAB_TPU_ADDR']
        TPU_ADDRESS = 'grpc://' + device_name
        #print('Found TPU at: {}'.format(TPU_ADDRESS))
        FLAGS.use_tpu = 1
    except KeyError:
        FLAGS.use_tpu = -1
        #print('TPU not found')

    logdir = os.path.join(FLAGS.model_log_dir)
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    gpus = tf.config.experimental.list_logical_devices("GPU")
    strategy = None
    if FLAGS.use_tpu == 1:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_ADDRESS)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
        strategy.experimental_enable_dynamic_batch_size = False

    elif len(gpus) > 1: # multiple GPUs in one VM
        strategy = tf.distribute.MirroredStrategy()

    else: # default strategy that works on CPU and single GPU
        strategy = tf.distribute.get_strategy()

    with strategy.scope():

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        initial_epoch = 0
        model = None
        if FLAGS.load_model_epoch == 0:
            # Get the SSD network and its anchors.
            model = models_factory.get_model(FLAGS.model_name, FLAGS.num_classes, input_shape=input_shape, softmax=softmax)
        else:
            # load the checkpoint from disk
            print("[INFO] loading {}... this model contains network used for training".format(model_path))
            model = load_model(model_path.format(epoch=FLAGS.load_model_epoch), compile=False)
            # update the learning rate
            print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
            K.set_value(model.optimizer.lr, FLAGS.learning_rate)
            print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))
            initial_epoch = FLAGS.load_model_epoch

        # Get model summary
        model.summary()

        # While saving the model store the model ssd_params.pkl in a pickle file for reloading the model
        if FLAGS.load_model_epoch == 0:
            with open(os.path.join(model_dir, 'ssd_params.pkl'), 'wb') as fw:
                print("model dictionary======================>>>>>>",type(model.additional_params), model.additional_params)
                fw.write(pickle.dumps(model.additional_params))
        else:
            with open(os.path.join(model_dir, 'ssd_params.pkl'), 'rb') as fr:
                model.additional_params = pickle.loads(fr.read())

        lr_callback = tf.keras.callbacks.LearningRateScheduler(step_decay)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        #optimizer = tf.keras.optimizers.SGD(lr=FLAGS.learning_rate, decay=1e-6, momentum=0.0, nesterov=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

        loss = SSDFocalLoss()
        # model.compile(optimizer=optimizer, loss=loss, metrics=loss.metrics)
        model.compile(optimizer=optimizer, loss=loss, sample_weight_mode='temporal')

        print("REPLICAS: ", strategy.num_replicas_in_sync)
        

        feat_shapes = model.additional_params['feat_shapes']
        anchor_ratios = model.additional_params['aspect_ratios']
        #print("aspect_ratios: ",anchor_ratios)
        anchor_sizes = model.additional_params['minmax_sizes']
        #print("minmax_sizes: ",anchor_sizes)
        anchor_steps = model.additional_params['steps']

        #print("steps: ",anchor_steps)
        special_ssd_boxes = model.additional_params['special_ssd_boxes']
        #print("special_ssd_boxes: ",special_ssd_boxes)
        anchor_offset = 0.5

        ssd_anchors = ssd_utils.anchors(
            input_shape, 
            feat_shapes,
            anchor_sizes,
            anchor_ratios,
            anchor_steps, 
            anchor_offset
        )

        preprocess_fn = get_preprocessing(FLAGS.preprocess_name, is_training=True)
        train_dataset = get_dataset(FLAGS.dataset_name, train_ref_path, ssd_anchors, FLAGS.num_classes, preprocess_fn=preprocess_fn, preprocess_fn_args={'out_shape': input_shape}, batch_size=FLAGS.batch_size)
        val_dataset = get_dataset(FLAGS.dataset_name, val_ref_path, ssd_anchors, FLAGS.num_classes, preprocess_fn=preprocess_fn, preprocess_fn_args={'out_shape': input_shape}, batch_size=FLAGS.batch_size)
        
        train_dataset = train_dataset.map(get_train_iter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.map(get_train_iter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        class_labels=config.whitelist_with_fields_label
        val_avt_callback_metrics = AvtMetricsCallback(val_dataset.take(val_steps_per_epoch), 'val', class_labels, batch_size=FLAGS.batch_size)
        #train_avt_callback_metrics = AvtMetricsCallback(train_dataset.take(int(num_batch_per_epoch*0.25)), 'train', dataset_meta.class_labels, dataset_meta.sub_catg_class_labels, BATCH_SIZE)
        
        append = False
        if FLAGS.load_model_epoch >0:
           append = True

        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(FLAGS.model_log_dir, 'training.csv'), append=append)


        model.fit(
            train_dataset,
            epochs=epochs,
            verbose=1, 
            shuffle=True,
            initial_epoch=initial_epoch,
            validation_data=val_dataset, 
            validation_steps=val_steps_per_epoch, 
            steps_per_epoch=steps_per_epoch,
            #max_queue_size=1, 
            workers=1,
            use_multiprocessing=False,
            callbacks=[
                                # Since we are explicit specifying progbarLogger, the builtin call back is disabled.
                                # Also this is a bug that ProgbarLogger does the average of metrics
                                # TODO: keras uses logs_dict to get metrics check->"AvtMetricsCallback", Bug ProgbarLogger does the update of state, hence disable progressbar using verbose=0
                                tf.keras.callbacks.ProgbarLogger(count_mode="steps", stateful_metrics=val_avt_callback_metrics.stateful_metrics),
                                tensorboard_callback,
                                val_avt_callback_metrics,
                                tf.keras.callbacks.ModelCheckpoint(
                                        filepath=model_path,
                                        monitor='val_average_macro_f1',
                                        mode='max',
                                        save_best_only=True,
                                        save_weights_only=False #Done: this is not working need to fix, I want it to be false for easy deployment
                                ),
                                tf.keras.callbacks.EarlyStopping(patience=20, monitor="val_loss"),
                                csv_logger]

        )




if __name__ == '__main__':
    tf.compat.v1.app.run()
