import tensorflow as tf
from utils import pp
from copy_task import *
from collections import namedtuple
def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

FLAGS = {
    'epoch': 1000,
    'input_dim': 5,
    'output_dim': 5,
    'length':4,
    'controller_layer_size':1,
    'write_head_size': 1,
    'read_head_size': 1,
    'test_max_length': 120,
    'checkpoint_dir': 'checkpoint'
}
FLAGS = convert(FLAGS)

with tf.device('/cpu:0'), tf.Session() as sess:
    cell, ntm = copy_train(FLAGS, sess)

