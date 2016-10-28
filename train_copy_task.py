import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16))
from utils import pp
from copy import *

flags = tf.app.flags
flags.DEFINE_string("task", "copy", "Task to run [copy, recall]")
flags.DEFINE_integer("epoch", 100000, "Epoch to train [100000]")
flags.DEFINE_integer("input_dim", 10, "Dimension of input [10]")
flags.DEFINE_integer("output_dim", 10, "Dimension of output [10]")
flags.DEFINE_integer("min_length", 1, "Minimum length of input sequence [1]")
flags.DEFINE_integer("max_length", 10, "Maximum length of output sequence [10]")
flags.DEFINE_integer("controller_layer_size", 1, "The size of LSTM controller [1]")
flags.DEFINE_integer("write_head_size", 1, "The number of write head [1]")
flags.DEFINE_integer("read_head_size", 1, "The number of read head [1]")
flags.DEFINE_integer("test_max_length", 120, "Maximum length of output sequence [120]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    with tf.device('/cpu:0'), tf.Session() as sess:
        cell, ntm = copy_train(FLAGS, sess)

        ntm.load(FLAGS.checkpoint_dir, 'copy')

        copy(ntm, FLAGS.test_max_length*1/3, sess)
        print
        copy(ntm, FLAGS.test_max_length*2/3, sess)
        print
        copy(ntm, FLAGS.test_max_length*3/3, sess)

if __name__ == '__main__':
    tf.app.run()
