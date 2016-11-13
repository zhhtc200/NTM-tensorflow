from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.seq2seq import sequence_loss

import ntm_cell

import os

class NTM(object):
    def __init__(self, cell, sess, length,
                 min_grad=-10, max_grad=+10,
                 lr=1e-4, momentum=0.9, decay=0.95,
                 scope="NTM"):
        """Create a neural turing machine specified by NTMCell "cell".

        Args:
            cell: An instantce of NTMCell.
            sess: A TensorFlow session.
            min_length: Minimum length of input sequence.
            max_length: Maximum length of input sequence for training.
            test_max_length: Maximum length of input sequence for testing.
            min_grad: (optional) Minimum gradient for gradient clipping [-10].
            max_grad: (optional) Maximum gradient for gradient clipping [+10].
            lr: (optional) Learning rate [1e-4].
            momentum: (optional) Momentum of RMSProp [0.9].
            decay: (optional) Decay rate of RMSProp [0.95].
        """
        if not isinstance(cell, ntm_cell.NTMCell):
            raise TypeError("cell must be an instance of NTMCell")

        self.cell = cell
        self.sess = sess
        self.scope = scope

        self.lr = lr
        self.momentum = momentum
        self.decay = decay

        self.min_grad = min_grad
        self.max_grad = max_grad
        self.length = length

        self.inputs = []
        self.outputs = []
        self.true_outputs = []
        self.masks = []

        self.prev_states = None
        self.input_states = defaultdict(list)
        self.output_states = defaultdict(list)

        self.losses = None
        self.optims = None
        self.grads = None

        self.saver = None
        self.params = None

        self.global_step = tf.Variable(0, trainable=False)
        self.opt = tf.train.RMSPropOptimizer(self.lr,
                                             decay=self.decay,
                                             momentum=self.momentum)
        self.build_model()

    def build_model(self):
        print(" [*] Building a NTM model")

        with tf.variable_scope(self.scope):
            prev_state = None
            for seq_length in range(2 * self.length + 2):
                # Build input
                input_ = tf.placeholder(tf.float32, [self.cell.input_dim], name='input_%s' % seq_length)
                true_output = tf.placeholder(tf.float32, [self.cell.output_dim], name='true_output_%s' % seq_length)
                mask = tf.placeholder(tf.float32, [1], name='mask%s' % seq_length)
                self.inputs.append(input_)
                self.true_outputs.append(true_output)
                self.masks.append(mask)

                if prev_state is None:
                    output, prev_state = self.cell(input_, state=None)
                else:
                    tf.get_variable_scope().reuse_variables()
                    output, prev_state = self.cell(input_, prev_state)
                self.outputs.append(output)

            print(" [*] Process to loss function.")
            self.losses = sequence_loss(logits=self.outputs,
                                        targets=self.true_outputs,
                                        weights=self.masks,
                                        average_across_timesteps=True,
                                        average_across_batch=False,
                                        softmax_loss_function=tf.nn.softmax_cross_entropy_with_logits)

            self.params = tf.trainable_variables()

            print(" [*] Process to gradients")
            grads = []
            for grad in tf.gradients(self.losses, self.params):
                if grad is not None:
                    grads.append(tf.clip_by_value(grad,
                                                  self.min_grad,
                                                  self.max_grad))
                else:
                    grads.append(grad)

            self.grads = grads
            self.optims = self.opt.apply_gradients(
                zip(grads, self.params),
                global_step=self.global_step)

        self.saver = tf.train.Saver()
        print(" [*] Build a NTM model finished")

    def save(self, checkpoint_dir, task_name, step):
        task_dir = os.path.join(checkpoint_dir, "%s_%s" % (task_name, self.length))
        file_name = "NTM_%s.model" % task_name

        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

        self.saver.save(self.sess,
                        os.path.join(task_dir, file_name),
                        global_step=step)

    def load(self, checkpoint_dir, task_name):
        print(" [*] Reading checkpoints...")

        task_dir = "%s_%s" % (task_name, self.length)
        checkpoint_dir = os.path.join(checkpoint_dir, task_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        else:
            raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)
