import tensorflow as tf


def self_attention(inputs, return_alphas=False, project=False, reuse=None, name="self_attention"):
    """Dimension reduction self-attention, e.g.: 3d inputs -> 2d outputs
    cf. Neural Machine Translation by Jointly Learning to Align and Translate (https://arxiv.org/pdf/1409.0473.pdf)
    :param inputs: input tensor
    :param return_alphas: if True, return alphas
    :param project: project input tensors
    :param reuse: share variables, reuse if True or tf.AUTO_REUSE, not reuse if None or False
    :param name: name
    :return: self attention result and alphas if return_alphas is True
    """
    with tf.variable_scope(name, reuse=reuse, dtype=tf.float32):
        hidden_size = inputs.shape[-1].value
        if project:
            # batch_size x seq_len x dim
            x = tf.layers.dense(inputs, units=hidden_size, use_bias=False, activation=tf.nn.tanh)
        else:
            x = inputs  # batch_size x seq_len x dim
        weight = tf.get_variable(name="weight", shape=[hidden_size, 1], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=0.01, seed=1227))
        x = tf.tensordot(x, weight, axes=1)  # batch_size x seq_len x 1
        alphas = tf.nn.softmax(x, axis=-2)  # batch_size x seq_len x 1
        output = tf.matmul(tf.transpose(inputs, perm=[0, 2, 1]), alphas)  # batch_size x dim x 1
        output = tf.squeeze(output, axis=-1)  # batch_size x dim
        if return_alphas:
            return output, alphas
        else:
            return output
