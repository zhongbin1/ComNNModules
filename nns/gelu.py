import numpy as np
import tensorflow as tf


def gelu(inputs):
    """Gaussian Error Linear Unit (Smoother version of ReLU)
    cf. https://arxiv.org/abs/1606.08415
    :param inputs: float tensor to perform activation.
    :return: `inputs` with GELU applied.
    """
    cdf = 0.5 * (1.0 + tf.nn.tanh((np.sqrt(2.0 / np.pi) * (inputs + 0.044715 * tf.pow(inputs, 3)))))
    return inputs * cdf
