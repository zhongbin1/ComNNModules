import tensorflow as tf


def focal_loss(logits, labels, seq_len, mask=None, weights=None, alpha=0.25, gamma=2):
    """Implementation of focal loss in Focal Loss for Dense Object Detection
    cf. http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
    cf. https://github.com/ailias/Focal-Loss-implement-on-Tensorflow/blob/master/focal_loss.py
    :param logits: predicted labels
    :param labels: ground truth labels
    :param seq_len: valid sequence length
    :param mask: mask, if None, use seq_len to build
    :param weights: weights
    :param alpha: alpha
    :param gamma: gamma
    :return: outputs
    """
    if mask is None:
        mask = tf.sequence_mask(seq_len, maxlen=tf.reduce_max(seq_len), dtype=tf.float32)
    if labels.get_shape().ndims < logits.get_shape().ndims:
        labels = tf.cast(tf.one_hot(labels, depth=logits.shape[-1].value, axis=-1), dtype=tf.float32)
    zeros = tf.zeros_like(logits, dtype=logits.dtype)
    if logits.shape[-1].value == 2:  # binary classification
        logits = tf.nn.sigmoid(logits)
        pos_logits_prob = tf.where(labels > zeros, labels - logits, zeros)
        neg_logits_prob = tf.where(labels > zeros, zeros, logits)
        cross_entropy = - alpha * (pos_logits_prob ** gamma) * tf.log(tf.clip_by_value(logits, 1e-8, 1.0)) \
                        - (1 - alpha) * (neg_logits_prob ** gamma) * tf.log(tf.clip_by_value(1.0 - logits, 1e-8, 1.0))
    else:
        logits = tf.nn.softmax(logits, axis=-1)
        logits_prob = tf.where(labels > zeros, labels - logits, zeros)
        cross_entropy = - alpha * (logits_prob ** gamma) * tf.log(tf.clip_by_value(logits, 1e-8, 1.0))
    if weights is not None:
        if weights.get_shape().ndims < logits.get_shape().ndims:
            weights = tf.expand_dims(weights, axis=-1)
        cross_entropy = cross_entropy * weights
    if mask is not None:
        if mask.get_shape().ndims < logits.get_shape().ndims:
            mask = tf.expand_dims(mask, axis=-1)
        cross_entropy = cross_entropy * mask
    return tf.reduce_sum(cross_entropy)