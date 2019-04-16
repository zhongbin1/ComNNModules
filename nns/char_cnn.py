import numpy as np
import tensorflow as tf


def char_cnn(inputs, kernels, filters, padding="VALID", activation=tf.nn.relu, reuse=None, name="char_cnn"):
    """Character-Aware Neural Language Models (http://arxiv.org/abs/1508.06615v4) for characters representation
        cf. https://github.com/allenai/bilm-tf/blob/master/bilm/model.py
        :param inputs: input tensor, normally with shape = (batch_size, max_sent_len, max_word_len, emb_dim)
        :param kernels: list of kernel (height of the CNN filter)
        :param filters: list of filter size (depth of each CNN layer)
        :param padding: padding approach
        :param activation: activation function, default is tanh
        :param reuse: share variables, reuse if True or tf.AUTO_REUSE, not reuse if None or False
        :param name: name
        :return: outputs
    """
    with tf.variable_scope(name, reuse=reuse):
        *_, max_chars, dim = inputs.get_shape().as_list()
        outputs = []
        for i, (kernel, filter_num) in enumerate(zip(kernels, filters)):
            if activation.__name__ == "relu":
                weight_init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
            else:  # normally, "tanh"
                weight_init = tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(1.0 / (kernel * dim)))
            weight = tf.get_variable(name="weight_%d" % i, shape=[1, kernel, dim, filter_num], initializer=weight_init,
                                     dtype=tf.float32)
            bias = tf.get_variable(name="bias_%d" % i, shape=[filter_num], initializer=tf.constant_initializer(0.0),
                                   dtype=tf.float32)
            conv = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding=padding, name="conv_%d" % i) + bias
            if max_chars is None:  # used for dynamic word length
                max_pool = tf.reduce_max(conv, axis=2, keepdims=True, name="max_pool_%d" % i)
            else:  # used for fixed word length
                max_pool = tf.nn.max_pool(conv, ksize=[1, 1, max_chars - kernel + 1, 1], strides=[1, 1, 1, 1],
                                          padding=padding, name="max_pool_%d" % i)
            conv = activation(max_pool)
            conv = tf.squeeze(conv, squeeze_dims=[2])
            outputs.append(conv)
        return tf.concat(values=outputs, axis=-1)  # batch_size x seq_len x sum(filters)
