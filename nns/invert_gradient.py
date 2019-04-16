import tensorflow as tf
from tensorflow.python.framework import ops


class InvertGradientBuilder:
    """
    cf. Domain-Adversarial Training of Neural Networks (https://arxiv.org/pdf/1505.07818v4.pdf)
    cf. https://github.com/pumpikano/tf-dann
    """
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, lw=1.0):
        grad_name = "InvertGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * lw]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


invert_gradient = InvertGradientBuilder()

'''
Usage:
given a computed tensor: feature
gradient reversal is applied as:
feat = invert_gradient(feature, lw)  #  this will flip the gradient when back-propagating through this operation
'''
