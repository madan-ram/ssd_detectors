"""SSD training utils."""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn import metrics as skm
from sklearn.metrics import precision_recall_fscore_support
import os
from utils.training import smooth_l1_loss, softmax_loss, focal_loss
from utils.training import plot_log

class AvtMetricsCallback(tf.keras.callbacks.Callback):
        """Keras callback to calculate metrics of a classifier for each epoch.
        Attributes
        ----------
        dataset
                The dataset
        """
        def __init__(self, dataset, dataset_type, class_label, batch_size, stateful_metrics=None):
                # stateful_metrics: Iterable of string names of metrics that
                #  should *not* be averaged over an epoch.
                #  Metrics in this list will be logged as-is in `on_epoch_end`.
                #  All others will be averaged in `on_epoch_end`.
                self.dataset = dataset
                self.dataset_type = dataset_type
                self.class_label = class_label
                self.batch_size = batch_size
                self.stateful_metrics = stateful_metrics or []
                self.stateful_metrics.append(self.dataset_type+'_average_micro_precision_for_class')
                self.stateful_metrics.append(self.dataset_type+'_average_micro_recall_for_class')
                self.stateful_metrics.append(self.dataset_type+'_average_micro_f1_for_class')

                self.stateful_metrics.append(self.dataset_type+'_class_wise_report')
                # self.stateful_metrics.append(self.dataset_type+'_average_macro_f1_for_class')
                # self.stateful_metrics.append(self.dataset_type+'_average_macro_f1')


        def on_epoch_end(self, epoch, logs={}):
                # y.shape (batches, priors, 4 x bbox_offset + n x class_label)
                print("Metrics analysis on ", self.dataset_type, 'dataset')

                logs = logs or {}

                class_true_label_lst = []
                class_pred_label_lst = []
                for index, (x, conf_true) in enumerate(self.dataset):
                        batch_size = tf.shape(conf_true)[0]
                        num_priors = tf.shape(conf_true)[1]
                        num_classes = tf.shape(conf_true)[2] - 4
                        eps = K.epsilon()
                        
                        conf_pred = self.model.predict(x)
                        
                        # confidence loss
                        conf_true = tf.reshape(conf_true[:, :, 4:], [-1, num_priors, num_classes])
                        conf_pred = tf.reshape(conf_pred[:, :, 4:], [-1, num_priors, num_classes])
                        
                        BACKGROUND_CLASS_ID = 0  # Choose the class of interest
                        class_true_label_lst += tf.reshape(tf.argmax(conf_true, axis=-1), [-1]).numpy().tolist()
                        class_pred_label_lst += tf.reshape(tf.argmax(conf_pred, axis=-1), [-1]).numpy().tolist()

                        # neg_mask = K.cast(K.equal(class_true, BACKGROUND_CLASS_ID), tf.float32)
                        
                        # pos_mask = tf.logical_not(neg_mask)
                        # pos_mask_float = tf.cast(pos_mask, tf.float32)
                        # num_pos = tf.reduce_sum(pos_mask_float)
                        
                        # offset loss
                        # loc_true = tf.reshape(y_true[:, :, 0:4], [-1, num_priors, 4])
                        # loc_pred = tf.reshape(y_pred[:, :, 0:4], [-1, num_priors, 4])
                        
                        # loc_loss = smooth_l1_loss(loc_true, loc_pred)
                        # pos_loc_loss = tf.reduce_sum(loc_loss * pos_mask_float) # only for positive ground truth
                        # loc_loss = pos_loc_loss/(num_pos + eps)
                        
                        # # total loss
                        # total_loss = conf_loss + self.alpha * loc_loss
                        
                        # return total_loss

                # class_gt_lst = []
                # sub_catg_class_gt_lst = []
                # class_p_lst = []
                # sub_catg_class_p_lst = []

                average_micro_precision_for_class = skm.precision_score(class_true_label_lst, class_pred_label_lst, average='micro')
                logs[self.dataset_type+'_average_micro_precision_for_class'] = average_micro_precision_for_class

                average_micro_recall_for_class = skm.recall_score(class_true_label_lst, class_pred_label_lst, average='micro')
                logs[self.dataset_type+'_average_micro_recall_for_class'] = average_micro_recall_for_class

                
                class_f1 = skm.f1_score(class_true_label_lst, class_pred_label_lst, average='micro')
                logs[self.dataset_type+'_average_micro_f1_for_class'] = class_f1



                # # Calculate macro f1 score
                class_f1 = skm.f1_score(class_true_label_lst, class_pred_label_lst, average='macro')
                logs[self.dataset_type+'_average_macro_f1'] = class_f1
                
                records = precision_recall_fscore_support(class_true_label_lst, class_pred_label_lst, labels=range(len(self.class_label)), average=None, zero_division=0)
                records = np.array(records).T.tolist()
                for cl, record in zip(self.class_label, records):
                        headers = ["precision", "recall", "f1-score", "support"]
                        for h, val in zip(headers, record):
                                metric_dict_path = os.path.join(self.dataset_type+'_class_wise_report', cl, h)
                                logs[metric_dict_path] = val
                                self.stateful_metrics.append(metric_dict_path)



                record = precision_recall_fscore_support(class_true_label_lst, class_pred_label_lst, labels=range(len(self.class_label)), average='macro', zero_division=0)
                headers = ["precision", "recall", "f1-score", "support"]
                for h, val in zip(headers, record):
                        metric_dict_path = os.path.join(self.dataset_type+'_class_wise_report', 'macro avg', h)
                        logs[metric_dict_path] = val
                        self.stateful_metrics.append(metric_dict_path)

class SSDFocalLoss(tf.keras.losses.Loss):
        def __init__(self, alpha=1.0, name='ssd_focal_loss', reduction=tf.keras.losses.Reduction.SUM, class_weights=None, gamma=2.0):
                #self.lambda_conf = lambda_conf
                #self.lambda_offsets = lambda_offsets
                self.name = name
                self.reduction = reduction
                self.alpha = alpha
                self.gamma = gamma

                # TODO: implement class weights
                self.class_weights = class_weights
                # build a lookup table
                if self.class_weights is not None:
                        self.class_weights_lookup_tensor = tf.lookup.StaticHashTable(
                                initializer=tf.lookup.KeyValueTensorInitializer(
                                        keys=tf.constant(list(self.class_weights.keys()), dtype=tf.int64),
                                        values=tf.constant(list(self.class_weights.values()), dtype=K.floatx())
                                ),
                                default_value=tf.constant(1.0),
                                name="class_weight"
                        )
        
        def __call__(self, y_true, y_pred, sample_weight=None):
                # y.shape (batches, priors, 4 x bbox_offset + n x class_label)
                
                batch_size = tf.shape(y_true)[0]
                num_priors = tf.shape(y_true)[1]
                num_classes = tf.shape(y_true)[2] - 4
                eps = K.epsilon()
                
                # confidence loss
                conf_true = tf.reshape(y_true[:, :, 4:], [-1, num_priors, num_classes])
                conf_pred = tf.reshape(y_pred[:, :, 4:], [-1, num_priors, num_classes])
                conf_loss = focal_loss(conf_true, conf_pred, gamma=self.gamma)

                # TODO: add class weights to this loss function
                num_total = num_priors*batch_size
                num_total = tf.cast(num_total, tf.float32)
                conf_loss = tf.reduce_sum(conf_loss)/(num_total + eps)
                
                # conf_loss = tf.reduce_sum(conf_loss)
                BACKGROUND_CLASS_ID = 0  # Choose the class of interest
                class_true = tf.argmax(conf_true, axis=-1)
                neg_mask = K.cast(K.equal(class_true, BACKGROUND_CLASS_ID), tf.bool)
                pos_mask = tf.logical_not(neg_mask)
                pos_mask_float = tf.cast(pos_mask, tf.float32)
                num_pos = tf.reduce_sum(pos_mask_float)
                
                # offset loss
                loc_true = tf.reshape(y_true[:, :, 0:4], [-1, num_priors, 4])
                loc_pred = tf.reshape(y_pred[:, :, 0:4], [-1, num_priors, 4])
                
                loc_loss = smooth_l1_loss(loc_true, loc_pred)
                pos_loc_loss = tf.reduce_sum(loc_loss * pos_mask_float) # only for positive ground truth
                loc_loss = pos_loc_loss/(num_pos + eps)
                
                # total loss
                total_loss = conf_loss + self.alpha * loc_loss
                
                return total_loss

