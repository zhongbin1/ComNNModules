import tensorflow as tf


def layer_norm(inputs, epsilon=1e-8, reuse=None, name="layer_norm"):
    """Layer Normalization
    cf. https://arxiv.org/abs/1607.06450
    cf. https://github.com/Kyubyong/transformer/blob/master/modules.py
    :param inputs: input tensor
    :param epsilon: epsilon
    :param reuse: if reuse
    :param name: name
    :return: normalized tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, axes=[-1], keep_dims=True)
        beta = tf.get_variable(name="beta", shape=params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable(name="gamma", shape=params_shape, initializer=tf.zeros_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = gamma * normalized + beta
        return outputs


def layer_norm_official(inputs, reuse=None, name="layer_norm"):
    """Layer Normalization (tensorflow contrib)
    cf. https://arxiv.org/abs/1607.06450
    cf. https://github.com/google-research/bert/blob/master/modeling.py
    :param inputs: input tensor
    :param reuse: if reuse
    :param name: name
    :return: normalized tensor
    """
    return tf.contrib.layers.layer_norm(inputs=inputs,
                                        center=True,
                                        scale=True,
                                        activation_fn=None,
                                        reuse=reuse,
                                        variables_collections=None,
                                        outputs_collections=None,
                                        trainable=True,
                                        begin_norm_axis=-1,
                                        begin_params_axis=-1,
                                        scope=name)
