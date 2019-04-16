import tensorflow as tf


def highway_network(inputs, num_unit, num_layer, activation=tf.nn.tanh, use_bias=True, reuse=None, name="highway_net"):
    """Highway Network (http://arxiv.org/abs/1505.00387)
        t = sigmoid(Wy + b)
        z = t * g(Wy + b) + (1 - t) * y
        where g is non-linearity, t is transform gate, and (1 - t) is carry gate.
        :param inputs: input tensor
        :param num_unit: number of neurons in highway network
        :param num_layer: number of layers in highway network
        :param activation: activation function of non-linearity computation, default tf.nn.relu
        :param use_bias: use bias or not
        :param reuse: share variables, reuse if True or tf.AUTO_REUSE, not reuse if None or False
        :param name: name
        :return: outputs
    """
    with tf.variable_scope(name, reuse=reuse):
        outputs = inputs
        for layer in range(num_layer):
            trans_gate = tf.layers.dense(outputs, units=num_unit, activation=tf.sigmoid, use_bias=use_bias,
                                         name="trans_gate_%d" % layer)
            hidden_outputs = tf.layers.dense(outputs, units=num_unit, activation=activation, use_bias=use_bias,
                                             name="hidden_%d" % layer)
            carry_gate = tf.subtract(1.0, trans_gate, name="carry_gate_%d" % layer)
            outputs = tf.add(tf.multiply(hidden_outputs, trans_gate), tf.multiply(outputs, carry_gate),
                             name="outputs_%d" % layer)
        return outputs
