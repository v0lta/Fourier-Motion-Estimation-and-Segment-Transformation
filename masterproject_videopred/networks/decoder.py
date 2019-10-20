from . import ops
import tensorflow as tf
from tensorflow.contrib import rnn

class TemporalCompressDecoder:
    """ decoder part of the model with temporal compression."""
    def __init__(self, cells, compress_flags, output_length, compress_rate=2):
        """
        Args:
            cells: list of RNNCells, one for each layer.
            compress_flags: list of 0, 1; 0 means without split.
            output_length: the length of the desired outputs.
        """
        self._cells = cells
        self._compress_flags = compress_flags
        self._output_length = output_length
        self._compress_rate = compress_rate

    def __call__(self, states, scope=None):
        """ get the outputs of the decoder.

        Args:
            states: a list of states from the encoder.

        Returns:
            a output tensor.
        """
        with tf.variable_scope(scope or "decoder"):
        #TODO: add argument kernel for upsample
            #get the shape of last state
            batch_size = states[-1].shape[0].value
            h = states[-1].shape[1].value
            w = states[-1].shape[2].value
            layers = len(self._cells)
            #get the steps for the first layer of encoder
            #comp_num: the number of cells that will split the outputs
            comp_num = self._compress_flags.count(1)
            if self._output_length % pow(self._compress_rate, comp_num) != 0:
                raise ValueError("the compress rate %d is not compatible with \
                                 length %d" % (self._compress_rate,
                                               self._output_length))
            steps = self._output_length/pow(self._compress_rate, comp_num)
            #TODO: how to deal with no inputs situation
            #in order to combine with states along last dimension
            zero_inputs = tf.zeros([steps, batch_size, h, w, 1])
            cur_inp = zero_inputs
            for i, (cell, flag) in enumerate(
                    zip(self._cells, self._compress_flags)):
                with tf.variable_scope("cell_%d" % i):
                    cur_inp, _ = tf.nn.dynamic_rnn(cell, cur_inp,
                                    initial_state=states[layers- i- 1],
                                    dtype=tf.float32, time_major=True)
                    #split first and then upsample to save the memory
                    #decompress temporally with flag equals 1
                    if flag == 1:
                        cur_inp = ops.split_(cur_inp, self._compress_rate)
                    #upsample each layers except the last one
                    if i < layers - 1:
                        cur_inp = ops.upsample(cur_inp, [3, 3])
            #return cur_inp
            #ensure that the output values are in [-1, 1]
            return tf.sigmoid(cur_inp)
            

class SimpleDecoder:
    """ decoder part of the model."""
    def __init__(self, cells, output_length):
        """
        Args:
            cells: list of RNNCells, one for each layer.
            output_length: the length of the desired outputs.
        """
        self._cells = cells
        self._output_length = output_length

    def __call__(self, states, scope=None):
        """ get the outputs of the decoder.

        Args:
            states: list of states of the encoder.

        Returns:
            a output tensor.
        """
        with tf.variable_scope(scope or "decoder"):
            #multi_cell = rnn.MultiRNNCell(self._cells)
            #get the shape of last state
            batch_size = states[-1].shape[0].value
            h = states[-1].shape[1].value
            w = states[-1].shape[2].value
            layers = len(self._cells)
            #get the steps for the layers of decoder
            steps = self._output_length
            #TODO: how to deal with no inputs situation
            #in order to combine with states along last dimension
            zero_inputs = tf.zeros([steps, batch_size, h, w, 1])
            cur_inp = zero_inputs
            for i, cell in enumerate(self._cells):
                with tf.variable_scope("cell_%d" % i):
                    cur_inp, _ = tf.nn.dynamic_rnn(cell, cur_inp,
                                    initial_state=states[layers- i- 1],
                                    dtype=tf.float32, time_major=True)
                    #upsample each layers except the last one                
                    if i < layers - 1:
                        cur_inp = ops.upsample(cur_inp, [3, 3])
            #return cur_inp
            #ensure that the output values are in [-1, 1]
            return tf.sigmoid(cur_inp)
