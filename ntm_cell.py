from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import reduce

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops
from ops import *


class NTMCell(object):
    def __init__(self, input_dim, output_dim,
                 mem_size=128, mem_dim=20, controller_dim=100,
                 controller_layer_size=1, shift_range=1,
                 write_head_size=1, read_head_size=1):
        """Initialize the parameters for an NTM cell.
        Args:
            input_dim: int, The number of units in the LSTM cell
            output_dim: int, The dimensionality of the inputs into the LSTM cell
            mem_size: (optional) int, The size of memory [128]
            mem_dim: (optional) int, The dimensionality for memory [20]
            controller_dim: (optional) int, The dimensionality for controller [100]
            controller_layer_size: (optional) int, The size of controller layer [1]
        """
        # initialize configs
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.controller_dim = controller_dim
        self.controller_layer_size = controller_layer_size
        self.shift_range = shift_range
        self.write_head_size = write_head_size
        self.read_head_size = read_head_size

        self.controller_collection = []
        for layer_idx in range(self.controller_layer_size):
            self.controller_collection.append(
                tf.nn.rnn_cell.LSTMCell(self.controller_dim)
            )

        self.states = []

    def __call__(self, input_, state=None, scope=None):
        """Run one step of NTM.

        Args:
            inputs: input Tensor, 2D, 1 x input_size.
            state: state Dictionary which contains M, read_w, write_w, read,
                output, hidden.
            scope: VariableScope for the created subgraph; defaults to class name.

        Returns:
            A tuple containing:
            - A 2D, batch x output_dim, Tensor representing the output of the LSTM
                after reading "input_" when previous state was "state".
                Here output_dim is:
                     num_proj if num_proj was set,
                     num_units otherwise.
            - A 2D, batch x state_size, Tensor representing the new state of LSTM
                after reading "input_" when previous state was "state".
        """
        if state == None:
            state = self.initial_state()

        M_prev = state['M']
        read_w_list_prev = state['read_w']
        write_w_list_prev = state['write_w']
        read_list_prev = state['read']
        hidden_list_prev = state['hidden']

        # build a controller
        last_output, hidden_list = \
            self.build_controller(input_, read_list_prev, hidden_list_prev)

        # build a memory
        M, read_w_list, write_w_list, read_list = \
            self.build_memory(M_prev,
                              read_w_list_prev,
                              write_w_list_prev,
                              last_output)

        # get a new output
        last_output = tf.reshape(last_output, [-1])
        new_output = self.new_output(last_output)

        state = {
            'M': M,
            'read_w': read_w_list,
            'write_w': write_w_list,
            'read': read_list,
            'hidden': hidden_list,
        }

        self.states.append(state)

        return new_output, state

    def new_output(self, output):
        """Logistic sigmoid output layers."""

        with tf.variable_scope('output'):
            return tf.sigmoid(
                linear(output, self.output_dim, bias=True,
                       scope='output')
            )

    def build_controller(self, input_, read_list_prev, hidden_list_prev):
        """Build LSTM controller."""

        with tf.variable_scope("controller"):
            input_ = tf.reshape(input_, [1, -1])
            cell_input = [input_] + read_list_prev
            cell_input = tf.concat(1, cell_input)

            hidden_list = list()
            for layer_idx in range(self.controller_layer_size):
                if layer_idx == 0:
                    layer_output, hidden = self.controller_collection[layer_idx](
                        inputs=cell_input,
                        state=hidden_list_prev[layer_idx],
                        scope='controller_%s' % layer_idx
                    )
                else:
                    output, hidden = self.controller_collection[layer_idx](
                        inputs=layer_output,
                        state=hidden_list_prev[layer_idx],
                        scope='controller_%s' % layer_idx
                    )

                hidden_list.append(hidden)

            return layer_output, hidden_list

    def build_memory(self, M_prev, read_w_list_prev, write_w_list_prev, last_output):
        """Build a memory to read & write."""

        with tf.variable_scope("memory"):
            # 3.1 Reading
            if self.read_head_size == 1:
                read_w_prev = read_w_list_prev[0]
                # print(read_w_prev)
                read_w, read = self.build_read_head(M_prev, read_w_prev, last_output, 0)
                # print(read_w)
                read_w_list = [read_w]

                read_list = [read]
            else:
                read_w_list = []
                read_list = []
                for idx in range(self.read_head_size):
                    read_w_prev_idx = read_w_list_prev[idx]
                    read_w_idx, read_idx = self.build_read_head(M_prev, read_w_prev_idx, last_output, idx)
                    read_w_list.append(read_w_idx)
                    read_list.append(read_idx)

            # 3.2 Writing
            if self.write_head_size == 1:
                write_w_prev = write_w_list_prev[0]
                write_w, write, erase = self.build_write_head(M_prev, write_w_prev, last_output, 0)
                # TODO: finalize this
                write_w_list = [write_w]
                write_w = tf.stack([write_w], axis=2)
                erase = tf.stack([erase], axis=1)
                write = tf.stack([write], axis=1)
                M_erase = tf.ones([self.mem_size, self.mem_dim]) - tf.matmul(write_w, erase)
                M_write = tf.matmul(write_w, write)
            else:
                write_w_list = []
                write_list = []
                erase_list = []

                M_erases = []
                M_writes = []

                for idx in range(self.write_head_size):
                    write_w_prev_idx = write_w_list_prev[idx]
                    write_w_idx, write_idx, erase_idx = self.build_write_head(
                        M_prev, write_w_prev_idx, last_output, idx
                    )
                    #write_w_list.append(tf.transpose(write_w_idx))
                    write_w_list.append(write_w_idx)
                    write_list.append(write_idx)
                    erase_list.append(erase_idx)
                    write_w_idx = tf.stack([write_w_idx], axis=2)
                    write_idx = tf.stack([write_idx], axis=1)
                    erase_idx = tf.stack([erase_idx], axis=1)
                    M_erases.append(tf.ones([1, self.mem_size, self.mem_dim]) - outer_product(write_w_idx, erase_idx))
                    M_writes.append(outer_product(write_w_idx, write_idx))

                M_erase = reduce(lambda x, y: x * y, M_erases)
                M_write = tf.add_n(M_writes)

            M = M_prev * M_erase + M_write

            return M, read_w_list, write_w_list, read_list

    def build_read_head(self, M_prev, read_w_prev, last_output, idx):
        return self.build_head(M_prev, read_w_prev, last_output, True, idx)

    def build_write_head(self, M_prev, write_w_prev, last_output, idx):
        return self.build_head(M_prev, write_w_prev, last_output, False, idx)

    def build_head(self, M_prev, w_prev, last_output, is_read, idx):
        scope = "read" if is_read else "write"

        with tf.variable_scope(scope):
            # Figure 2.
            # Amplify or attenuate the precision
            with tf.variable_scope("k"):
                k = tf.contrib.layers.fully_connected(
                    inputs=last_output, num_outputs=self.mem_dim,
                    activation_fn=None, scope='k_%s' % idx,
                    weights_initializer=tf.truncated_normal_initializer(),
                    biases_initializer=tf.truncated_normal_initializer()
                )
                # k = tf.tanh(linear(last_output, self.mem_dim, bias=True, scope='k_%s' % idx))
            # Interpolation gate
            with tf.variable_scope("g"):
                g = tf.contrib.layers.fully_connected(
                    inputs=last_output, num_outputs=1,
                    activation_fn=None, scope='g_%s' % idx,
                    weights_initializer=tf.truncated_normal_initializer(),
                    biases_initializer=tf.truncated_normal_initializer()
                )
                # g = tf.sigmoid(linear(last_output, 1, bias=True, scope='g_%s' % idx))
            # shift weighting
            with tf.variable_scope("s_w"):
                s_w = tf.contrib.layers.fully_connected(
                    inputs=last_output, num_outputs=2 * self.shift_range + 1,
                    activation_fn=tf.nn.softmax, scope='s_w_%s' % idx,
                    weights_initializer=tf.truncated_normal_initializer(),
                    biases_initializer=tf.truncated_normal_initializer()
                )
                # w = linear(last_output, 2 * self.shift_range + 1, bias=True, scope='s_w_%s' % idx)
            with tf.variable_scope("beta"):
                beta = tf.contrib.layers.fully_connected(
                    inputs=last_output, num_outputs=1,
                    activation_fn=tf.nn.softplus, scope='beta_%s' % idx,
                    weights_initializer=tf.truncated_normal_initializer(),
                    biases_initializer=tf.truncated_normal_initializer()
                )  # [batch_size x 1]
                # beta = tf.nn.softplus(linear(last_output, 1, bias=True, scope='beta_%s' % idx))
            with tf.variable_scope("gamma"):
                gamma = tf.contrib.layers.fully_connected(
                    inputs=last_output, num_outputs=1,
                    activation_fn=tf.nn.softplus, scope='gamma_%s' % idx,
                    weights_initializer=tf.truncated_normal_initializer(),
                    biases_initializer=tf.truncated_normal_initializer()
                )
                gamma = tf.add(gamma, tf.constant(1.0))
                # gamma = tf.add(tf.nn.softplus(linear(last_output, 1, bias=True, scope='gamma_%s' % idx)),
                #                tf.constant(1.0))

            # 3.3.1
            # Cosine similarity
            # print(M_prev.get_shape())
            # print(k.get_shape())
            similarity = smooth_cosine_similarity(M_prev, k)  # [batch_size x men_size]
            # Focusing by content
            content_focused_w = tf.nn.softmax(similarity * beta)  # [batch_size x men_size]

            # 3.3.2
            # Focusing by location
            gated_w = tf.add_n([
                (content_focused_w * g),
                (w_prev * (tf.constant(1.0) - g))
            ])  # [batch_size x men_size]

            # Convolutional shifts
            conv_w = circular_convolution(gated_w, s_w)
            # Sharpening
            powed_conv_w = tf.pow(conv_w, gamma)
            w = powed_conv_w / tf.reduce_sum(powed_conv_w, axis=1)

            if is_read:
                # 3.1 Reading
                read = tf.matmul(tf.stack([w], axis=1), M_prev)
                read = tf.squeeze(read, axis=1)
                return w, read
            else:
                # 3.2 Writing
                erase = tf.contrib.layers.fully_connected(
                    inputs=last_output, num_outputs=self.mem_dim,
                    activation_fn=tf.nn.sigmoid, scope='erase_%s' % idx,
                    weights_initializer=tf.truncated_normal_initializer(),
                    biases_initializer=tf.truncated_normal_initializer()
                )
                add = tf.contrib.layers.fully_connected(
                    inputs=last_output, num_outputs=self.mem_dim,
                    activation_fn=tf.nn.sigmoid, scope='add_%s' % idx,
                    weights_initializer=tf.truncated_normal_initializer(),
                    biases_initializer=tf.truncated_normal_initializer()
                )
                # erase = tf.sigmoid(
                #     linear(last_output, self.mem_dim, bias=True, scope='erase_%s' % idx))
                # add = tf.tanh(linear(last_output, self.mem_dim,
                #                      bias=True, scope='add_%s' % idx))
                return w, add, erase

    def initial_state(self):
        self.states = []
        with tf.variable_scope("init_cell"):
            # memory
            M_init = array_ops.zeros([1, self.mem_size, self.mem_dim], dtype=tf.float32)

            # read weights
            read_w_list_init = []
            read_list_init = []
            for idx in range(self.read_head_size):
                read_w_idx = array_ops.zeros([1, self.mem_size],
                                             dtype=tf.float32,
                                             name='read_w_%d' % idx)
                read_w_list_init.append(softmax(read_w_idx))

                read_init_idx = array_ops.zeros([1, self.mem_dim],
                                                dtype=tf.float32,
                                                name='read_init_%d' % idx)
                read_list_init.append(read_init_idx)

            # write weights
            write_w_list_init = []
            for idx in range(self.write_head_size):
                write_w_idx = array_ops.zeros([1, self.mem_size],
                                              dtype=tf.float32,
                                              name='write_w_%s' % idx)
                write_w_list_init.append(softmax(write_w_idx))

            # controller state
            hidden_init_list = []
            for idx in range(self.controller_layer_size):
                output_init_idx = array_ops.zeros(
                    [1, self.controller_dim],
                    dtype=tf.float32,
                    name='output_init_%s' % idx
                )
                hidden_init_idx = array_ops.zeros(
                    [1, self.controller_dim],
                    dtype=tf.float32,
                    name='hidden_init_%s' % idx
                )
                hidden_init_list.append((hidden_init_idx, output_init_idx))

            state = {
                'M': M_init,
                'read_w': read_w_list_init,
                'write_w': write_w_list_init,
                'read': read_list_init,
                'hidden': hidden_init_list
            }

            self.states.append(state)

            return state


if __name__ == "__main__":
    cell = NTMCell(input_dim=5,
                   output_dim=5,
                   controller_layer_size=5,
                   write_head_size=1,
                   read_head_size=1,
                   controller_dim=32)
    prev_state = None
    with tf.variable_scope('debug'):
        for seq_length in range(3):
            input_ = tf.zeros([5], dtype=tf.float32)
            if prev_state is None:
                output, prev_state = cell(input_, state=None)
            else:
                tf.get_variable_scope().reuse_variables()
                output, prev_state = cell(input_, prev_state)
