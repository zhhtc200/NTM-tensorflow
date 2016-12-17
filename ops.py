import math
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs

from utils import *


def linear(args, output_size, bias, bias_dev=0.5, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = []
    for a in args:
        try:
            shapes.append(a.get_shape().as_list())
        except Exception as e:
            shapes.append(a.shape)

    is_vector = False
    for idx, shape in enumerate(shapes):
        if len(shape) != 2:
            is_vector = True
            args[idx] = tf.reshape(args[idx], [1, -1])
            total_arg_size += shape[0]
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        matrix = vs.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable(
            "Bias", [output_size],
            initializer=init_ops.truncated_normal_initializer(stddev=bias_dev))

    if is_vector:
        return tf.reshape(res + bias_term, [-1])
    else:
        return res + bias_term


def smooth_cosine_similarity(m, v):
    """Computes smooth cosine similarity.

    Args:
        m: a 2-D `Tensor` [batch_size, mem_loc, men_size]
        v: a 2-D `Tensor` [batch_size, men_size]
    """
    # shape_x = m.get_shape().as_list()
    # shape_y = v.get_shape().as_list()
    # if shape_x[1] != shape_y[0]:
    #     raise ValueError("Smooth cosine similarity is expecting same dimemsnion")

    m_norm = tf.sqrt(tf.reduce_sum(tf.pow(m, 2), 2)) #[batch_size, mem_loc]
    v_norm = tf.sqrt(tf.reduce_sum(tf.pow(v, 2), 1)) #[batch_size, 1]
    m_dot_v = tf.matmul(m, tf.expand_dims(v, 2))
    m_dot_v = tf.squeeze(m_dot_v, axis=2) #[batch_size, mem_loc]

    similarity = tf.div(tf.div(m_dot_v, m_norm + 1e-4), v_norm + 1e-4)
    #similarity = tf.div(tf.reshape(m_dot_v, [-1]), m_norm * v_norm + 1e-3)
    return similarity


def circular_convolution(v, k):
    """Computes circular convolution.

    Args:
        v: a 1-D `Tensor` (vector)
        k: a 1-D `Tensor` (kernel)
    """
    batch_size = int(v.get_shape()[0])
    size = int(v.get_shape()[1])
    kernel_size = int(k.get_shape()[1])
    kernel_shift = int(math.floor(kernel_size/2.0))

    def loop(idx):
        if idx < 0: return size + idx
        if idx >= size: return idx - size
        else: return idx

    output = []
    for batch_id in range(batch_size):
        kernels = []
        for i in range(size):
            indices = [loop(i+j) for j in range(kernel_shift, -kernel_shift-1, -1)]
            v_ = tf.gather(v[batch_id], indices)
            kernels.append(tf.reduce_sum(v_ * k[batch_id], 0))
        batch_output = tf.pack(kernels)
        output.append(batch_output)
    output = tf.pack(output)

    # # code with double loop
    # for i in xrange(size):
    #     for j in xrange(kernel_size):
    #         idx = i + kernel_shift - j + 1
    #         if idx < 0: idx = idx + size
    #         if idx >= size: idx = idx - size
    #         w = tf.gather(v, int(idx)) * tf.gather(kernel, j)
    #         output = tf.scatter_add(output, [i], tf.reshape(w, [1, -1]))

    return output

def outer_product(*inputs):
    """Computes outer product.

    Args:
        inputs: a list of 1-D `Tensor` (vector)
    """
    inputs = list(inputs)
    order = len(inputs)

    for idx, input_ in enumerate(inputs):
        if len(input_.get_shape()) == 1:
            inputs[idx] = tf.reshape(input_, [-1, 1] if idx % 2 == 0 else [1, -1])

    if order == 2:
        output = tf.mul(inputs[0], inputs[1])
    elif order == 3:
        size = []
        idx = 1
        for i in range(order):
            size.append(inputs[i].get_shape()[0])
        output = tf.zeros(size)

        u, v, w = inputs[0], inputs[1], inputs[2]
        uv = tf.mul(inputs[0], inputs[1])
        for i in range(self.size[-1]):
            output = tf.scatter_add(output, [0, 0, i], uv)

    return output





def scalar_mul(x, beta, name=None):
    return x * beta


def scalar_div(x, beta, name=None):
    return x / beta
