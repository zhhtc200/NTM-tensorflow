import os
import time
import numpy as np
import tensorflow as tf
from random import randint

from ntm import NTM
from utils import pprint
from ntm_cell import NTMCell

print_interval = 50

def copy_train(config, sess):
    # delimiter flag for start and end
    start_symbol = np.zeros([config.input_dim], dtype=np.float32)
    start_symbol[0] = 1
    end_symbol = np.zeros([config.input_dim], dtype=np.float32)
    end_symbol[1] = 1

    cell = NTMCell(input_dim=config.input_dim,
                   output_dim=config.output_dim,
                   controller_layer_size=config.controller_layer_size,
                   write_head_size=config.write_head_size,
                   read_head_size=config.read_head_size)
    ntm = NTM(cell, sess, config.length)

    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)


        print(" [*] Initialize all variables")
        tf.global_variable_initializer().run()
        print(" [*] Initialization finished")
    else:
        ntm.load(config.checkpoint_dir, 'copy')

    start_time = time.time()
    print('')
    for idx in range(config.epoch):
        seq_length = np.random.randint(2, config.length+1)
        X, Y, masks = build_seq_batch(seq_length, config.length, config.input_dim - 2)

        feed_dict = {input_: vec for vec, input_ in zip(X, ntm.inputs)}
        feed_dict.update(
            {true_output: vec for vec, true_output in zip(Y, ntm.true_outputs)}
        )
        feed_dict.update(
            {mask: vec for vec, mask in zip(masks, ntm.masks)}
        )

        if idx % print_interval != 0:
            _, cost, step = sess.run([ntm.optims,
                                      ntm.losses,
                                      ntm.global_step], feed_dict=feed_dict)
        else:
            _, cost, step, Y_pre = sess.run([ntm.optims,
                                      ntm.losses,
                                      ntm.global_step,
                                      ntm.outputs], feed_dict=feed_dict)
            print("[%5d] %2d: %.4f (%.1fs)" \
                  % (idx, seq_length, cost, time.time() - start_time), end='')
            Y_pre = np.array(Y_pre)
            mask_id = masks.reshape(-1).astype(bool)
            print(np.argmax(Y, axis=1)[mask_id], end='')
            print(np.argmax(Y_pre, axis=1)[mask_id])

    print("Training Copy task finished")
    ntm.save(config.checkpoint_dir, 'copy', idx)
    return cell, ntm


def build_seq_batch(length, max_len, input_dim):
    X_input = np.zeros((max_len * 2 + 2, input_dim + 2))
    Y_input = np.zeros((max_len * 2 + 2, input_dim + 2))
    Mask    = np.zeros((max_len * 2 + 2, 1))

    # Build time sequences
    X_input[0, 0] = 1
    for time_instant in range(length):
        picked = np.random.randint(input_dim)
        X_input[time_instant+1, picked+2] = 1
        Y_input[time_instant + length + 2, picked + 2] = 1
    X_input[length + 1, 1] = 1
    Mask[length + 2:2 * length + 2, 0] = 1

    return X_input, Y_input, Mask
