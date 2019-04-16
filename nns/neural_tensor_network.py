import tensorflow as tf


def neural_tensor_network(inputs1, inputs2, k, activation=tf.nn.tanh, compute_score=False, reuse=None,
                          name="neural_tensor_network"):
    """Neural Tensor Network
    cf. paper: https://cs.stanford.edu/~danqi/papers/nips2013.pdf
    cf. repo: https://github.com/dddoss/tensorflow-socher-ntn/blob/master/code/ntn.py
    :param inputs1: input tensor with shape = (batch_size, dim1)
    :param inputs2: another input tensor with shape = (batch_size, dim2), normally, dim1 == dim2
    :param k:  number of slices
    :param activation: activation function, default tanh
    :param compute_score: if True, return the computed score, otherwise return the output tensor
    :param reuse: share variables, reuse if True or tf.AUTO_REUSE, not reuse if None or False
    :param name: name scope
    :return: computed score (if compute_score is True) or tensor (otherwise)
    """
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        dim1 = inputs1.shape[-1].value
        dim2 = inputs2.shape[-1].value
        tensor_weight = tf.get_variable(name="tensor_weight", shape=[k, dim1, dim2], dtype=tf.float32)
        weight = tf.get_variable(name="weight", shape=[dim1 + dim2, k], dtype=tf.float32)
        bias = tf.get_variable(name="bias", shape=[k], dtype=tf.float32, initializer=tf.zeros_initializer())
        # computation
        tensor_product = list()
        for i in range(k):
            tensor_product.append(tf.reduce_sum(tf.matmul(inputs1, tensor_weight[i]) * inputs2, axis=1))
        tensor_product = tf.reshape(tf.concat(tensor_product, axis=0), shape=[-1, k])
        standard_product = tf.matmul(tf.concat([inputs1, inputs2], axis=-1), weight)
        preactivation = tensor_product + standard_product + bias
        output = activation(preactivation)  # shape = (batch_size, k)
        if compute_score:
            u = tf.Variable(tf.ones[1, k], name="u", trainable=True)
            output = tf.matmul(output, tf.transpose(u, [1, 0]))
            return output  # shape = (batch_size, )
        else:
            return output  # shape = (batch_size, k)
