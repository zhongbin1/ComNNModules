import math
import tensorflow as tf


def label_smoothing(inputs, epsilon=0.1):
    dim = inputs.get_shape().as_list()
    return (1 - epsilon) * inputs + (epsilon / dim)


def weight_initializer(stddev=0.02):
    return tf.truncated_normal_initializer(mean=0.0, stddev=stddev)


def layer_norm(inputs, epsilon=1e-8, reuse=None, name="layer_norm"):
    """Layer Normalization
    cf. https://arxiv.org/abs/1607.06450
    cf. https://github.com/Kyubyong/transformer/blob/master/modules.py
    TensorFlow official implementation:
        `tf.contrib.layers.layer_norm(inputs=inputs, begin_norm_axis=-1, begin_params_axis=-1, scope=name)`
    :param inputs: input tensor
    :param epsilon: epsilon
    :param reuse: reuse
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


def dropout(inputs, drop_rate=0.0, name="dropout"):
    with tf.variable_scope(name):
        if drop_rate is None or drop_rate == 0.0:
            return inputs
        else:
            return tf.nn.dropout(inputs, keep_prob=1.0 - drop_rate)


def get_shape(inputs):
    shape = inputs.shape.as_list()
    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)
    if not non_static_indexes:
        return shape
    else:
        dyn_shape = tf.shape(inputs)
        for index in non_static_indexes:
            shape[index] = dyn_shape[index]
        return shape


def split_heads(inputs, num_heads):
    """
    :param inputs: tensor with shape [batch_size, seq_len, dim]
    :param num_heads: number of heads
    :return: tensor with shape [batch_size, num_heads, seq_len, head_size]
    """
    old_shape = get_shape(inputs)  # [batch_size, seq_len, dim]
    new_shape = old_shape[:-1] + [num_heads, old_shape[-1] // num_heads]
    x = tf.reshape(inputs, shape=new_shape)  # batch_size x seq_len x num_heads x head_size
    return tf.transpose(x, perm=[0, 2, 1, 3])  # batch_size x num_heads x seq_len x head_size


def combine_heads(inputs):
    """
    :param inputs: tensor with shape [batch_size, num_heads, seq_len, head_size]
    :return: tensor with shape [batch_size, seq_len, dim]
    """
    x = tf.transpose(inputs, perm=[0, 2, 1, 3])  # batch_size x seq_len x num_heads x head_size
    old_shape = get_shape(x)
    new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
    return tf.reshape(x, shape=new_shape)  # batch_size x seq_len x dim


def create_attention_mask(from_tensor, to_mask):
    """
    cf. https://github.com/google-research/bert/blob/master/modeling.py
    :param from_tensor: 2D or 3D tensor with shape [batch_size, from_seq_len, ...]
    :param to_mask: int32 tensor with shape [batch_size, to_seq_len]
    :return: tensor with shape [batch_size, from_seq_len, to_seq_len]
    """
    from_shape = get_shape(from_tensor)
    to_shape = get_shape(to_mask)
    batch_size = from_shape[0]
    from_seq_len = from_shape[1]
    to_seq_len = to_shape[1]

    to_mask = tf.cast(tf.reshape(to_mask, shape=[batch_size, 1, to_seq_len]), dtype=tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We don't actually care if we attend *from*
    # padding tokens (only *to* padding tokens) so we create a tensor of all ones.
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast = tf.ones(shape=[batch_size, from_seq_len, 1], dtype=tf.float32)

    return broadcast * to_mask


def create_attention_mask_from_seq_len(from_tensor, to_seq_lengths, max_len=None):
    """
    :param from_tensor: 2D or 3D tensor with shape [batch_size, from_seq_len, ...]
    :param to_seq_lengths: int32 tensor with shape [batch_size], each value indicates a sequence length
    :param max_len: the maximal sequence length of to_seq_lengths, if None, use the maximal value of to_seq_lengths
    :return: tensor with shape [batch_size, from_seq_len, to_seq_len]
    """
    from_shape = get_shape(from_tensor)
    batch_size = from_shape[0]
    from_seq_len = from_shape[1]
    if max_len is None:
        max_len = tf.reduce_max(to_seq_lengths)
    to_seq_len = max_len

    to_mask = tf.sequence_mask(to_seq_lengths, maxlen=max_len, dtype=tf.float32)
    to_mask = tf.reshape(to_mask, shape=[batch_size, 1, to_seq_len])

    broadcast = tf.ones(shape=[batch_size, from_seq_len, 1], dtype=tf.float32)

    return broadcast * to_mask


def multihead_attention(inputs1, inputs2, attention_mask=None, num_heads=1, head_size=512, attention_drop_rate=0.0,
                        hidden_drop_rate=0.0, query_act=None, key_act=None, value_act=None, kernel_init=0.02,
                        reuse=None, name="multihead_attention"):
    """Multihead attention
    :param inputs1: input tensor with shape [batch_size, seq_len_1, dim_1]
    :param inputs2: input tensor with shape [batch_size, seq_len_2, dim_2] (for self-attention, input1 == inputs2)
    :param attention_mask: int32 tensor of with shape [batch_size, seq_len_1, seq_len_2]
    :param num_heads: number of heads
    :param head_size: head size
    :param attention_drop_rate:
    :param hidden_drop_rate:
    :param query_act: query activation function
    :param key_act: key activation function
    :param value_act: value activation function
    :param kernel_init: standard deviation of truncated normal initializer
    :param reuse: reuse
    :param name: name
    :return: attention outputs
    """
    with tf.variable_scope(name, reuse=reuse):
        # feed forward layer
        query = tf.layers.dense(inputs1, units=num_heads * head_size, activation=query_act, name="query",
                                kernel_initializer=weight_initializer(stddev=kernel_init))
        key = tf.layers.dense(inputs2, units=num_heads * head_size, activation=key_act, name="key",
                              kernel_initializer=weight_initializer(stddev=kernel_init))
        value = tf.layers.dense(inputs2, units=num_heads * head_size, activation=value_act, name="value",
                                kernel_initializer=weight_initializer(stddev=kernel_init))
        # split heads
        query = split_heads(query, num_heads=num_heads)  # [batch_size, num_heads, seq_len_1, head_size]
        key = split_heads(key, num_heads=num_heads)  # [batch_size, num_heads, seq_len_2, head_size]
        value = split_heads(value, num_heads=num_heads)  # [batch_size, num_heads, seq_len_2, head_size]
        # attention
        attention_scores = tf.matmul(query, key, transpose_b=True)  # [batch_size, num_heads, seq_len_1, seq_len_2]
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(head_size)))
        if attention_mask is not None:  # add bias
            attention_mask = tf.expand_dims(attention_mask, axis=[1])  # [batch_size, 1, seq_len_1, seq_len_2]
            bias = (1.0 - tf.cast(attention_mask, dtype=tf.float32)) * (-2 ** 32 + 1)
            attention_scores += bias
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)  # [batch_size, num_heads, seq_len_1, seq_len_2]
        attention_probs = dropout(attention_probs, drop_rate=attention_drop_rate)
        # compute context
        context = tf.matmul(attention_probs, value)  # [batch_size, num_heads, seq_len_1, head_size]
        context = combine_heads(context)  # [batch_size, seq_len_1, num_heads * head_size]
        # feed forward layer
        context = tf.layers.dense(context, units=num_heads * head_size, name="output",
                                  kernel_initializer=weight_initializer(stddev=kernel_init))
        context = dropout(context, drop_rate=hidden_drop_rate, name="attention_dropout")
        return context


def transformer(inputs, seq_len=None, attention_mask=None, num_units=768, num_layers=12, num_heads=12,
                intermediate=True, intermediate_units=3072, intermediate_act=None, hidden_drop_rate=0.1,
                attention_drop_rate=0.1, kernel_init=0.02, return_all_layers=False, reuse=None, name="transformer"):
    """Transformer encoder block
    cf. https://github.com/google-research/bert/blob/master/modeling.py
    cf. https://github.com/tensorflow/models/tree/master/official/transformer/model
    cf. https://github.com/Kyubyong/transformer/blob/master/modules.py
    :param inputs: input tensor with shape [batch_size, seq_len, dim]
    :param seq_len: input tensor with shape [batch_size], which indicates the valid sequence length of inputs
    :param attention_mask: int32 tensor of with shape [batch_size, seq_len, seq_len]
    :param num_units: hidden size of transformer
    :param num_layers: number of layers in transformer
    :param num_heads: number of heads for multihead attention
    :param intermediate: boolean, if use an intermediate dense layer between multihead attention and output dense layer
    :param intermediate_units: hidden size of intermediate dense layer
    :param intermediate_act: activation function of intermediate dense layer
    :param hidden_drop_rate: dropout rate for hidden layers
    :param attention_drop_rate: dropout rate for multihead attention
    :param kernel_init: standard deviation of truncated normal initializer
    :param return_all_layers: whether to return outputs of all layers
    :param reuse: reuse
    :param name: name
    :return: outputs
    """
    assert num_units % num_heads == 0, "The num_units (%d) must be multiple of num_heads (%d)" % (num_units, num_heads)
    with tf.variable_scope(name, reuse=reuse):
        head_size = int(num_units / num_heads)
        # create attention mask
        if attention_mask is None and seq_len is not None:
            attention_mask = create_attention_mask_from_seq_len(inputs, seq_len)
        # project inputs to make sure that the input last dimension is equal to num_units
        if inputs.get_shape().as_list()[-1] != num_units:
            inputs = tf.layers.dense(inputs, units=num_units, use_bias=False, activation=None, name="input_projection")
        # n-layer transformer block
        prev_output = inputs
        all_layer_outputs = []
        for layer in range(num_layers):
            with tf.variable_scope("layer_%d" % layer):
                layer_input = prev_output
                # multihead attention
                attention_output = multihead_attention(inputs1=layer_input, inputs2=layer_input,
                                                       attention_mask=attention_mask, num_heads=num_heads,
                                                       head_size=head_size, attention_drop_rate=attention_drop_rate,
                                                       hidden_drop_rate=hidden_drop_rate, kernel_init=kernel_init,
                                                       name="multihead_attention")
                # residual connection and layer normalization
                attention_output = layer_norm(attention_output + layer_input, name="layer_norm_attention")

                # add intermediate dense layer
                if intermediate:
                    attention_output = tf.layers.dense(attention_output, units=intermediate_units,
                                                       activation=intermediate_act, name="intermediate_layer",
                                                       kernel_initializer=weight_initializer(kernel_init))

                # feed forward layer
                layer_output = tf.layers.dense(attention_output, units=num_units, name="output",
                                               kernel_initializer=weight_initializer(kernel_init))
                layer_output = dropout(layer_output, drop_rate=hidden_drop_rate, name="output_dropout")
                layer_output = layer_norm(layer_output + attention_output, name="layer_norm_output")
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

        # return encoded results
        if return_all_layers:
            return all_layer_outputs
        else:
            return prev_output
