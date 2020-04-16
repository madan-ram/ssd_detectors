"""TextBoxes++ training utils."""

import tensorflow as tf
import tensorflow.keras.backend as K
from utils.training import smooth_l1_loss, focal_loss

# def compute_metrics(class_true, class_pred, conf, top_k=100):
#     """Compute precision, recall, accuracy and f-measure for top_k predictions.
    
#     from top_k predictions that are TP FN or FP (TN kept out)
#     """
    
#     # TODO: does this only work for one class?

#     top_k = tf.cast(top_k, tf.int32)
#     eps = K.epsilon()
    
#     mask = tf.greater(class_true + class_pred, 0)
#     #mask = tf.logical_or(tf.greater(class_true, 0), tf.greater(class_pred, 0))
#     mask_float = tf.cast(mask, tf.float32)
    
#     vals, idxs = tf.nn.top_k(conf * mask_float, k=top_k)
    
#     top_k_class_true = tf.gather(class_true, idxs)
#     top_k_class_pred = tf.gather(class_pred, idxs)
    
#     true_mask = tf.equal(top_k_class_true, top_k_class_pred)
#     false_mask = tf.logical_not(true_mask)
#     pos_mask = tf.greater(top_k_class_pred, 0)
#     neg_mask = tf.logical_not(pos_mask)
    
#     tp = tf.reduce_sum(input_tensor=tf.cast(tf.logical_and(true_mask, pos_mask), tf.float32))
#     fp = tf.reduce_sum(input_tensor=tf.cast(tf.logical_and(false_mask, pos_mask), tf.float32))
#     fn = tf.reduce_sum(input_tensor=tf.cast(tf.logical_and(false_mask, neg_mask), tf.float32))
#     tn = tf.reduce_sum(input_tensor=tf.cast(tf.logical_and(true_mask, neg_mask), tf.float32))
    
#     precision = tp / (tp + fp + eps)
#     recall = tp / (tp + fn + eps)
#     accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
#     fmeasure = 2 * (precision * recall) / (precision + recall + eps)
    
#     return precision, recall, accuracy, fmeasure

# =========================================================================== #
# SSD loss function.
# =========================================================================== #
def ssd_losses(match_threshold=0.5, negative_ratio=3., alpha=1.,
       label_smoothing=0., num_classes=2,
       scope=None
   ):
    
    @tf.function
    def _ssd_losses(targets, logits):
        """Loss functions for training the SSD 300 VGG network.

        This function defines the different loss components of the SSD, and
        adds them to the TF loss collection.

        Arguments:
          logits: (list of) predictions logits Tensors;
          localisations: (list of) localisations Tensors;
          gclasses: (list of) groundtruth labels Tensors;
          glocalisations: (list of) groundtruth localisations Tensors;
          gscores: (list of) groundtruth score Tensors;
        """
        localisations, logits  = logits[:, :, 0:4], logits[:, :, 4:num_classes+4]
        localisations.set_shape((None, None, 4))
        logits.set_shape((None, None, num_classes))
        glocalisations, gclasses, gscores = targets[:, :, 0:4], targets[:, :, 4:5], targets[:, :, 5]

        print(logits, "=======================")
        print(localisations, "=======================")
        print(glocalisations, "=======================")
        print(gclasses, "=======================")
        print(gscores, "=======================")

        with tf.compat.v1.name_scope(scope, 'ssd_losses'):
            l_cross_pos = []
            l_cross_neg = []
            l_loc = []

            for i in tf.range(len(logits)):
                    dtype = logits[i].dtype
                # with tf.compat.v1.name_scope('block_%i' % i):
                    # Determine weights Tensor.
                    pmask = gscores[i] > match_threshold
                    fpmask = tf.cast(pmask, dtype)
                    n_positives = tf.reduce_sum(input_tensor=fpmask)

                    # Select some random negative entries.
                    # n_entries = np.prod(gclasses[i].get_shape().as_list())
                    # r_positive = n_positives / n_entries
                    # r_negative = negative_ratio * n_positives / (n_entries - n_positives)

                    # Negative mask.
                    no_classes = tf.cast(pmask, tf.int32)
                    predictions = tf.nn.softmax(logits[i])

                    print(predictions.get_shape(), '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

                    nmask = tf.logical_and(tf.logical_not(pmask),
                                           gscores[i] > -0.5)

                    fnmask = tf.cast(nmask, dtype)

                    print(nmask, fnmask, '*******************************')
                    nvalues = tf.compat.v1.where(nmask,
                                       predictions[:, :, :, :, 0],
                                       1. - fnmask)

                    nvalues_flat = tf.reshape(nvalues, [-1])
                    # Number of negative entries to select.
                    n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
                    n_neg = tf.maximum(n_neg, tf.size(input=nvalues_flat) // 8)
                    n_neg = tf.maximum(n_neg, tf.shape(input=nvalues)[0] * 4)
                    max_neg_entries = 1 + tf.cast(tf.reduce_sum(input_tensor=fnmask), tf.int32)
                    n_neg = tf.minimum(n_neg, max_neg_entries)

                    val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
                    minval = val[-1]
                    # Final negative mask.
                    nmask = tf.logical_and(nmask, -nvalues > minval)
                    fnmask = tf.cast(nmask, dtype)

                    # Add cross-entropy loss.
                    with tf.compat.v1.name_scope('cross_entropy_pos'):
                        # print('logits ->', logits[i].get_shape())
                        # print('gclasses ->', gclasses[i].get_shape())
                        # print("=================================================")
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i], labels=gclasses[i])
                        # loss = focal_loss_v2(logits[i], gclasses[i])
                        loss = tf.compat.v1.losses.compute_weighted_loss(loss, fpmask)
                        l_cross_pos.append(loss)

                    with tf.compat.v1.name_scope('cross_entropy_neg'):
                        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i], labels=no_classes)
                        # loss = focal_loss_v2(logits[i], no_classes)
                        loss = tf.compat.v1.losses.compute_weighted_loss(loss, fnmask)
                        l_cross_neg.append(loss)

                    # Add localization loss: smooth L1, L2, ...
                    with tf.compat.v1.name_scope('localization'):
                        # Weights Tensor: positive mask + random negative.
                        weights = tf.expand_dims(alpha * fpmask, axis=-1)
                        loss = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
                        loss = tf.compat.v1.losses.compute_weighted_loss(loss, weights)
                        l_loc.append(loss)

            # Additional total losses...
            with tf.compat.v1.name_scope('total'):
                total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
                total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
                total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
                total_loc = tf.add_n(l_loc, 'localization')

                # Add to EXTRA LOSSES TF.collection
                tf.compat.v1.add_to_collection('EXTRA_LOSSES', total_cross_pos)
                tf.compat.v1.add_to_collection('EXTRA_LOSSES', total_cross_neg)
                tf.compat.v1.add_to_collection('EXTRA_LOSSES', total_cross)
                tf.compat.v1.add_to_collection('EXTRA_LOSSES', total_loc)

            total_losses = tf.add_n([total_cross, total_loc], 'total_loss')

            print(total_losses, '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            return total_losses
    return _ssd_losses
