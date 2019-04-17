import tensorflow as tf


def gated_cnn(inputs, num_layers, filter_h, filter_w, filter_size, block_size, reuse=None, name="gated_cnn"):
    """Gated CNN
    cf. https://arxiv.org/pdf/1612.08083.pdf
    cf. https://github.com/anantzoid/Language-Modeling-GatedCNN
    cf. https://github.com/noowad/language-model-by-gated-cnn
    :param inputs: input tensor, for the language modeling, it should be the gathered word vectors
    :param num_layers: number of layers
    :param filter_h: kernel height
    :param filter_w: kernel width
    :param filter_size: filter size
    :param block_size: block size for residual connection
    :param reuse: if reuse
    :param name: name
    :return: outputs
    """
    with tf.variable_scope(name, reuse=reuse):
        batch_size, context_size, emb_dim = inputs.get_shape()
        inputs = tf.expand_dims(inputs, axis=-1)  # expand dimension
        res_inputs = inputs
        for layer in range(num_layers):
            # convolution
            dim = inputs.get_shape()[-1]
            filter_sz = filter_size if layer < num_layers - 1 else 1
            # compute linear
            weight_l = tf.get_variable(name="weight_l_%d" % layer, shape=(filter_h, filter_w, dim, filter_sz),
                                       dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
            bias_l = tf.get_variable(name="bias_l_%d" % layer, shape=[filter_sz], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
            conv_l = tf.add(tf.nn.conv2d(inputs, weight_l, strides=[1, 1, 1, 1], padding="SAME"), bias_l)
            # compute gated
            weight_g = tf.get_variable(name="weight_g_%d" % layer, shape=(filter_h, filter_w, dim, filter_sz),
                                       dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
            bias_g = tf.get_variable(name="bias_g_%d" % layer, shape=[filter_sz], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.0))
            conv_g = tf.add(tf.nn.conv2d(inputs, weight_g, strides=[1, 1, 1, 1], padding="SAME"), bias_g)
            # merge
            inputs = tf.multiply(conv_l, tf.sigmoid(conv_g))
            if layer % block_size == 0:
                inputs = tf.add(inputs, res_inputs)
                res_inputs = inputs
        inputs = tf.reshape(inputs, shape=[batch_size, context_size, emb_dim])
        return inputs


def gated_cnn_simple(inputs, mask, feature_size, kernel_size, reuse=None, name="gated_conv_net"):
    """Implementation of Language Modeling with Gated Convolutional Networks (https://arxiv.org/pdf/1612.08083.pdf)
    cf. https://github.com/anantzoid/Language-Modeling-GatedCNN/blob/master/model.py
    :param inputs: input tensor, with shape = (batch_size, time_sequence, dim)
    :param mask: 0-1 mask with shape = (batch_size, time_sequence), normally derived from sequence length
    :param feature_size: convolutional feature size
    :param kernel_size: convolutional kernel size
    :param reuse: share variables, reuse if True or tf.AUTO_REUSE, not reuse if None or False
    :param name: name
    :return: outputs
    """
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        num_channels = inputs.get_shape().as_list()[-1]
        weights = tf.get_variable(name="filter", shape=[kernel_size, num_channels, 2 * feature_size])
        bias = tf.get_variable(name="bias", shape=[2 * feature_size])
        outputs = tf.nn.bias_add(tf.nn.convolution(inputs, weights, padding="SAME"), bias)
        gate, linear = tf.split(outputs, num_or_size_splits=2, axis=-1)
        outputs = tf.nn.sigmoid(gate) * linear
        if mask is None:
            return outputs
        else:
            return outputs * mask[:, :, None]
