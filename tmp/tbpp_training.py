"""TextBoxes++ training utils."""

import tensorflow as tf
import tensorflow.keras.backend as K
from utils.training import smooth_l1_loss, focal_loss

def compute_metrics(class_true, class_pred, conf, top_k=100):
    """Compute precision, recall, accuracy and f-measure for top_k predictions.
    
    from top_k predictions that are TP FN or FP (TN kept out)
    """
    
    # TODO: does this only work for one class?

    top_k = tf.cast(top_k, tf.int32)
    eps = K.epsilon()
    
    mask = tf.greater(class_true + class_pred, 0)
    #mask = tf.logical_or(tf.greater(class_true, 0), tf.greater(class_pred, 0))
    mask_float = tf.cast(mask, tf.float32)
    
    vals, idxs = tf.nn.top_k(conf * mask_float, k=top_k)
    
    top_k_class_true = tf.gather(class_true, idxs)
    top_k_class_pred = tf.gather(class_pred, idxs)
    
    true_mask = tf.equal(top_k_class_true, top_k_class_pred)
    false_mask = tf.logical_not(true_mask)
    pos_mask = tf.greater(top_k_class_pred, 0)
    neg_mask = tf.logical_not(pos_mask)
    
    tp = tf.reduce_sum(input_tensor=tf.cast(tf.logical_and(true_mask, pos_mask), tf.float32))
    fp = tf.reduce_sum(input_tensor=tf.cast(tf.logical_and(false_mask, pos_mask), tf.float32))
    fn = tf.reduce_sum(input_tensor=tf.cast(tf.logical_and(false_mask, neg_mask), tf.float32))
    tn = tf.reduce_sum(input_tensor=tf.cast(tf.logical_and(true_mask, neg_mask), tf.float32))
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    fmeasure = 2 * (precision * recall) / (precision + recall + eps)
    
    return precision, recall, accuracy, fmeasure

class TBPPFocalLoss(object):

    def __init__(self, lambda_conf=1000.0, lambda_offsets=1.0):
        self.lambda_conf = lambda_conf
        self.lambda_offsets = lambda_offsets
        self.metrics = []
    
    def compute(self, y_true, y_pred):
        # y.shape (batches, priors, 4 x bbox_offset + 8 x quadrilaterals + 5 x rbbox_offsets + n x class_label)
        
        batch_size = tf.shape(input=y_true)[0]
        num_priors = tf.shape(input=y_true)[1]
        num_classes = tf.shape(input=y_true)[2] - 17
        eps = K.epsilon()
        
        # confidence loss
        conf_true = tf.reshape(y_true[:,:,17:], [-1, num_classes])
        conf_pred = tf.reshape(y_pred[:,:,17:], [-1, num_classes])
        
        class_true = tf.argmax(input=conf_true, axis=1)
        class_pred = tf.argmax(input=conf_pred, axis=1)
        conf = tf.reduce_max(input_tensor=conf_pred, axis=1)
        
        neg_mask_float = conf_true[:,0]
        neg_mask = tf.cast(neg_mask_float, tf.bool)
        pos_mask = tf.logical_not(neg_mask)
        pos_mask_float = tf.cast(pos_mask, tf.float32)
        num_total = tf.cast(tf.shape(input=conf_true)[0], tf.float32)
        num_pos = tf.reduce_sum(input_tensor=pos_mask_float)
        num_neg = num_total - num_pos
        
        conf_loss = focal_loss(conf_true, conf_pred, alpha=[0.002, 0.998])
        conf_loss = tf.reduce_sum(input_tensor=conf_loss)
        conf_loss = conf_loss / (num_total + eps)
        conf_loss = self.lambda_conf * conf_loss
        
        # offset loss, bbox, quadrilaterals, rbbox
        loc_true = tf.reshape(y_true[:,:,0:17], [-1, 17])
        loc_pred = tf.reshape(y_pred[:,:,0:17], [-1, 17])
        
        loc_loss = smooth_l1_loss(loc_true, loc_pred)
        pos_loc_loss = tf.reduce_sum(input_tensor=loc_loss * pos_mask_float) # only for positives
        loc_loss = pos_loc_loss / (num_pos + eps)
        loc_loss = self.lambda_offsets * loc_loss
        
        # total loss
        total_loss = conf_loss + loc_loss
        
        # metrics
        precision, recall, accuracy, fmeasure = compute_metrics(class_true, class_pred, conf, top_k=100*batch_size)
        
        def make_fcn(t):
            return lambda y_true, y_pred: t
        for name in ['conf_loss', 
                     'loc_loss', 
                     'precision', 
                     'recall',
                     'accuracy',
                     'fmeasure', 
                     'num_pos',
                     'num_neg'
                    ]:
            f = make_fcn(eval(name))
            f.__name__ = name
            self.metrics.append(f)
        
        return total_loss
