import math
import tensorflow as tf


def weight_initializer(stddev=0.02):
    return tf.truncated_normal_initializer(mean=0.0, stddev=stddev)


def dropout(inputs, drop_rate=0.0):
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


def multihead_attention(inputs1, inputs2, attention_mask=None, num_heads=1, head_size=512, attention_drop_rate=0.0,
                        hidden_drop_rate=0.0, query_act=None, key_act=None, value_act=None, kernel_init=0.02,
                        reuse=None, name="multihead_attention"):
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
            bias = (1.0 - tf.cast(attention_mask, dtype=tf.float32)) * 1e-9
            attention_scores += bias
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)  # [batch_size, num_heads, seq_len_1, seq_len_2]
        attention_probs = dropout(attention_probs, drop_rate=attention_drop_rate)
        # compute context
        context = tf.matmul(attention_probs, value)  # [batch_size, num_heads, seq_len_1, head_size]
        context = combine_heads(context)  # [batch_size, seq_len_1, num_heads * head_size]
        # feed forward layer
        context = tf.layers.dense(context, units=num_heads * head_size, name="context",
                                  kernel_initializer=weight_initializer(stddev=kernel_init))
        context = dropout(context, drop_rate=hidden_drop_rate)
        return context
