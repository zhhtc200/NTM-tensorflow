from __future__ import print_function
import os
import time
import numpy as np
import tensorflow as tf
from ntm import NTM
from ntm_cell import NTMCell

copy_times = 5


def build_seq_batch(length, max_len, input_dim):
    X_input = np.zeros((max_len * (copy_times + 1) + 2, input_dim + 2))
    Y_input = np.zeros((max_len * (copy_times + 1) + 2, input_dim + 2))
    Mask = np.zeros(max_len * (copy_times + 1) + 2)

    # Build time sequences
    X_input[0, input_dim] = 1
    for time_instant in range(length):
        picked = np.random.randint(input_dim)
        X_input[time_instant + 1, picked] = 1
        for copy_id in range(copy_times):
            Y_input[time_instant + (copy_id + 1) * length + 2, picked] = 1
    X_input[length + 1, input_dim + 1] = 1
    Mask[length + 2:(copy_times + 1) * length + 2] = 1

    return X_input, Y_input, Mask

if __name__ == "__main__":
    print_interval = 50
    config = {
        'epoch': 100000,
        'input_dim': 7,
        'output_dim': 7,
        'length': 5,
        'controller_layer_size': 1,
        'write_head_size': 1,
        'read_head_size': 1,
        'checkpoint_dir': 'checkpoint'
    }

    with tf.device('/cpu:0'), tf.Session() as sess:
        # delimiter flag for start and end
        cell = NTMCell(input_dim=config['input_dim'],
                       output_dim=config['output_dim'],
                       controller_layer_size=config['controller_layer_size'],
                       write_head_size=config['write_head_size'],
                       read_head_size=config['read_head_size'])
        ntm = NTM(cell, sess, config['length'] * (copy_times + 1) + 2)

        if not os.path.isdir(config['checkpoint_dir']+'/n_copy_'+str(config['length'] * (copy_times + 1) + 2)):
            print(" [*] Initialize all variables")
            tf.global_variables_initializer().run()
            print(" [*] Initialization finished")
        else:
            ntm.load(config['checkpoint_dir'], 'copy')

        start_time = time.time()
        print('')
        for idx in range(config['epoch']):
            seq_length = np.random.randint(2, config['length'] + 1)
            X, Y, masks = build_seq_batch(seq_length, config['length'], config['input_dim'] - 2)

            feed_dict = {ntm.inputs: X, ntm.true_outputs: Y, ntm.masks: masks}

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
        ntm.save(config['checkpoint_dir'], 'n_copy', idx)
