from . import ops
import tensorflow as tf
from tensorflow.contrib import rnn
class TemporalCompressEncoder:
    """ encoder part of the model."""
    def __init__(self, cells, compress_flags, compress_rate=2):
        """
        Args:
            cells: list of RNNCells, one for each layer.
            compress_flags: list of 0, 1; 0 for without compression.
            layers: the number of layers.
        """

        self._cells = cells
        #layers can induced from cells
        #self._layers = layers
        self._compress_flags = compress_flags
        self._compress_rate = compress_rate

    def __call__(self, inputs, scope=None):
        """
        Args:
            inputs: tensor of shape (time, batch_size, h, w, d)
        returns:
            states: a list of states, each one is the last state in each layer
        """
        #TODO: add argument kernel for downsample
        with tf.variable_scope(scope or "encoder"):
            layers = len(self._cells)
            cur_inp = inputs
            states = []
            for i, (cell, flag) in enumerate(
                    zip(self._cells, self._compress_flags)):
                with tf.variable_scope("cell_%d" % i):
                    cur_inp, state = tf.nn.dynamic_rnn(cell, cur_inp,
                                        dtype=tf.float32, time_major=True)
                    #spatial comression for all layers except the last one
                    if i < layers - 1:
                        cur_inp = ops.downsample(cur_inp, [3, 3])
                    #temporal compression for layers with compress flag being 1
                    if flag == 1:
                        cur_inp = ops.combine_(cur_inp, self._compress_rate)
                    states.append(state)
        return states

class SimpleEncoder:
    """ simple encoder without compressing temporally."""
    def __init__(self, cells):
        """
        Args:
            cells: list of RNNCells, one for each layer.
        """

        self._cells = cells

    def __call__(self, inputs, scope=None):
        """
        Args:
            inputs: tensor of shape (time, batch_size, h, w, d)
        returns:
            states:a list of states, each one is the last state in each layer
        """
        #TODO: add argument kernel for downsample
        with tf.variable_scope(scope or "encoder"):
            layers = len(self._cells)
            cur_inp = inputs
            states = []
            for i, cell in enumerate(self._cells):
                with tf.variable_scope("cell_%d" % i):
                    cur_inp, state = tf.nn.dynamic_rnn(cell, cur_inp,
                                        dtype=tf.float32, time_major=True)
                    #spatial comression for all layers except the last one
                    if i < layers - 1:
                        cur_inp = ops.downsample(cur_inp, [3, 3])
                    states.append(state)
        return states
