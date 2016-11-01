import os
import time
import numpy as np
import tensorflow as tf
from random import randint

from ntm import NTM
from utils import pprint
from ntm_cell import NTMCell

print_interval = 50


# def copy(ntm, seq_length, sess, print_=True):
#     start_symbol = np.zeros([ntm.cell.input_dim], dtype=np.float32)
#     start_symbol[0] = 1
#     end_symbol = np.zeros([ntm.cell.input_dim], dtype=np.float32)
#     end_symbol[1] = 1
#
#     seq = generate_copy_sequence(seq_length, ntm.cell.input_dim - 2)
#
#     feed_dict = {input_: vec for vec, input_ in zip(seq, ntm.inputs)}
#     feed_dict.update(
#         {true_output: vec for vec, true_output in zip(seq, ntm.true_outputs)}
#     )
#     feed_dict.update({
#         ntm.start_symbol: start_symbol,
#         ntm.end_symbol: end_symbol
#     })
#
#     input_states = [state['write_w'][0] for state in ntm.input_states[seq_length]]
#     output_states = [state['read_w'][0] for state in ntm.get_output_states(seq_length)]
#
#     result = sess.run(ntm.get_outputs(seq_length) + \
#                       input_states + output_states + \
#                       [ntm.get_loss(seq_length)],
#                       feed_dict=feed_dict)
#
#     is_sz = len(input_states)
#     os_sz = len(output_states)
#
#     outputs = result[:seq_length]
#     read_ws = result[seq_length:seq_length + is_sz]
#     write_ws = result[seq_length + is_sz:seq_length + is_sz + os_sz]
#     loss = result[-1]
#
#     if print_:
#         np.set_printoptions(suppress=True)
#         print(" true output : ")
#         pprint(seq)
#         print(" predicted output :")
#         pprint(np.round(outputs))
#         print(" Loss : %f" % loss)
#         np.set_printoptions(suppress=False)
#     else:
#         return seq, outputs, read_ws, write_ws, loss


def copy_train(config, sess):
    if not os.path.isdir(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

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

    print(" [*] Initialize all variables")
    tf.initialize_all_variables().run()
    print(" [*] Initialization finished")

    start_time = time.time()
    print('')
    for idx in range(config.epoch):
        seq_length = config.length
        X, Y, masks = build_seq_batch(seq_length, config.input_dim - 2)

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
            print("\r [%5d] %2d: %.4f (%.1fs)" \
                  % (idx, seq_length, cost, time.time() - start_time), end='')
            Y_pre = np.array(Y_pre)
            print(np.argmax(Y, axis=1)[-5:], end='')
            print(np.argmax(Y_pre, axis=1)[-5:], end='', flush=True)

    print("Training Copy task finished")
    return cell, ntm

def build_seq_batch(max_length, input_dim):
    X_input = np.zeros((max_length*2+2, input_dim+2))
    Y_input = np.zeros((max_length*2+2, input_dim+2))
    Mask    = np.zeros((max_length*2+2, 1))

    # Build time sequences
    X_input[0, 0] = 1
    for time_instant in range(max_length):
        picked = np.random.randint(input_dim)
        X_input[time_instant+1, picked+2] = 1
        Y_input[time_instant+max_length+2, picked+2] = 1
    X_input[max_length+1, 1] = 1
    Mask[max_length+2:2*max_length+2,0] = 1

    return X_input, Y_input, Mask
