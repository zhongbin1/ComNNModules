import tensorflow as tf


def densely_connect_rnn(inputs, seq_len, num_layers, num_units, return_state=False, reuse=None, name="densely_rnn"):
    """Densely Connected Bidirectional LSTM
    cf. https://arxiv.org/pdf/1802.00889.pdf
    :param inputs: input tensors, with shape = (batch_size, seq_len, dim)
    :param seq_len: sequence length
    :param num_layers: number of layers
    :param num_units: number of units for rnn cell, list type, tuple type or int type
    :param reuse: bool, if reuse the model
    :param return_state: if return the last hidden states
    :param name: name
    :return: learned features
    """
    if isinstance(num_units, list) or isinstance(num_units, tuple):
        assert num_layers == len(num_units), "the size of hidden units list must equal to num_layers"
    else:
        num_units = [num_units] * num_layers
    with tf.variable_scope(name, reuse=reuse):
        cur_outputs = inputs
        cur_state = None
        for layer in range(num_layers):
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=num_units[layer], name="cell_fw_%d" % layer)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=num_units[layer], name="cell_bw_%d" % layer)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, cur_outputs, seq_len, dtype=tf.float32)
            outputs = tf.concat(outputs, axis=-1)
            if layer < num_layers - 1:
                cur_outputs = tf.concat([cur_outputs, outputs], axis=-1)
            else:
                cur_outputs = outputs
                cur_state = tf.concat([states[0][1], states[1][1]], axis=-1)
        if return_state:
            return cur_outputs, cur_state
        else:
            return cur_outputs
