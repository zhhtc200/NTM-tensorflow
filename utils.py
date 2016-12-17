import tensorflow as tf

eps = 1e-12


def softmax(x):
    """Compute softmax.

    Args:
        x: a 2-D `Tensor` (matrix) or 1-D `Tensor` (vector)
    """
    try:
        return tf.nn.softmax(x + eps)
    except:
        return tf.reshape(tf.nn.softmax(tf.reshape(x + eps, [1, -1])), [-1])