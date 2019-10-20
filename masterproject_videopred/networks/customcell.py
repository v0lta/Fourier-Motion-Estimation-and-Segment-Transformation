import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from .ops import depthwise_warp
from .ops import simple_warp
from .ops import simple_fft_warp
#import pdb


class ConvGRU(RNNCell):
    """ implementation of convolutional GRU.
        reference: http://arxiv.org/abs/1706.03458
    """

    def __init__(self, kernel_size, depth, input_dims, output_depth=None,
                 strides=None, reuse=None, activation=tf.nn.tanh):
        """ initialize convolutional GRU.
        kernel_size: the size of the kernels
        depth: the kernel depth
        input_dims: [height, width] channel is of no interest
        output_depth: the depth of the output convolution
        strides: the step sizes with which the sliding is done
        activation: activation function of hidden state and output
        """
        self._kernel_size = kernel_size
        # use tf.cast will get a tensor, which is not accepted by int()
        # self._depth = tf.cast(depth, tf.int32)
        self._depth = depth
        self._input_dims = input_dims
        self._output_depth = output_depth
        if strides is None:
            self._strides = [1, 1]
        else:
            self._strides = strides
        self._reuse = reuse
        self._activation = activation

    @property
    def recurrent_size(self):
        """ shape of tensors flowing along the recurrent connections.
        """
        # the result with "same" padding
        return tf.TensorShape([np.ceil(self._input_dims[0] /
                                       self._strides[0]),
                               np.ceil(self._input_dims[1] /
                                       self._strides[1]), self._depth])

    @property
    def output_size(self):
        """ size of outputs produced by the cell.
        """
        if self._output_depth is None:
            return self.recurrent_size
        return tf.TensorShape([self.recurrent_size[0], self.recurrent_size[1],
                               self._output_depth])

    @property
    def state_size(self):
        """ size of the cell state.
        """
        return self.recurrent_size

    def __call__(self, inputs, state, scope=None):
        """
            Args:
                inputs: input of shape [batch_size, height, width, channels]
            Returns:
                outputs: with shape self.output_size
                state: with shape self.state_size
        """
        with tf.variable_scope(scope or str(type(self).__name__),
                               reuse=self._reuse):
            channels = inputs.shape[-1].value
            Wi = tf.get_variable('input_weights', (self._kernel_size
                                                   + [channels]
                                                   + [self._depth * 3]))
            Wr = tf.get_variable('recurrent_weights', (self._kernel_size
                                                       + [self._depth]
                                                       + [self._depth * 3]))
            b = tf.get_variable('state_bias', ([self.recurrent_size[0],
                                                self.recurrent_size[1]]
                                               + [self._depth * 3]),
                                initializer=tf.zeros_initializer())
            input_conv = tf.nn.convolution(inputs, Wi, 'SAME')
            linear = input_conv + b
            recurrent_conv = tf.nn.convolution(state, Wr, 'SAME')
            # z the update gate, r the reset gate, h the candidate state
            # linear is of shape [batch_size, recurrent_size], so the axis
            # for split will be
            z, r, h = tf.split(linear, 3, axis=self.recurrent_size.ndims)
            rz, rr, rh = tf.split(recurrent_conv, 3,
                                  axis=self.recurrent_size.ndims)
            z += rz
            r += rr
            z = tf.nn.sigmoid(z)
            r = tf.nn.sigmoid(r)
            h += r * rh
            h = self._activation(h)
            # apply dropout to the state candidate
            h = (1 - z) * state + z * h

            if self._output_depth is not None:
                Wproj = tf.get_variable('projection_weights',
                                        (self._kernel_size + [self._depth]
                                         + [self._output_depth]))
                out = tf.nn.convolution(h, Wproj, 'SAME')
            else:
                out = h

            if self._reuse is None:
                self._reuse = True

            return out, h


class StridedConvGRU(RNNCell):
    """Efficient reimplementation of a ConvGRU that can downsample on
    input or upsample on output.
    """

    def __init__(self, kernel_size, depth, input_dims, output_depth=None,
                 strides=None, transpose=False,
                 reuse=None, normalize=False, is_training=True, trainable=True,
                 constrain=False, activation=tf.nn.tanh):
        """ input_dims: [height, width, channels]
            kernel_size: The size of the kernels, which are slided over the image
                         [size_in_X, size_in_Y].
            strides: The step sizes with which the sliding is done (will lead to
                     downsampling if >1) [size_in_X, size_in_Y].
            depth: The kernel depth.
            transpose: if True, upsampling in output
            output_depth: The depth of the output convolution.
        """
        # super().__init__(_reuse=reuse)
        self.kernel_size = kernel_size
        self.depth = depth
        self.input_dims = [int(dim) for dim in input_dims]
        if strides is None:
            self.strides = [1, 1]
        else:
            self.strides = strides
        self.output_depth = output_depth
        self.transpose = transpose
        self.normalize = normalize
        self.is_training = is_training
        self.trainable = trainable
        if constrain is True:
            assert normalize is True
        self.constrained = constrain
        self.activation = activation
        self.reuse = reuse

    @property
    def recurrent_size(self):
        """ Shape of the tensors flowing along the recurrent connections.
        """
        if self.transpose is False:
            return tf.TensorShape([np.ceil(self.input_dims[0] / self.strides[0]),
                                   np.ceil(
                                       self.input_dims[1] / self.strides[1]),
                                   self.depth])
        else:
            return tf.TensorShape([self.input_dims[0],
                                   self.input_dims[1],
                                   self.depth])

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell.
        """
        if self.transpose is True:
            hw = [np.ceil(self.input_dims[0] * self.strides[0]),
                  np.ceil(self.input_dims[1] * self.strides[1])]
        else:
            hw = [int(self.recurrent_size[0]),
                  int(self.recurrent_size[1])]
        if self.output_depth is None:
            depth = self.depth
        else:
            depth = self.output_depth
        return tf.TensorShape(hw + [depth])

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        It can be represented by an Integer,
        a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        return self.recurrent_size

    def __call__(self, x, state, scope=None):
        """
          Args:
            x: Input tensor of shape [batch_size, height, width, channels]
          Returns:
            outputs: shape self.output_size
            state: shape self.state_size
        """
        input_shape = tf.Tensor.get_shape(x)
        num_channels = int(input_shape[-1])
        batch_size = x.shape[0].value

        h = state
        with tf.variable_scope(scope, default_name=str(type(self).__name__),
                               reuse=self.reuse):
            depth_f = 3

            if self.normalize:
                # normalization according to:
                # Weight Normalization: A Simple Reparameterization to Accelerate
                # Training of Deep Neural Networks by Salimans, Tim
                # Kingma, Diederik P.
                vr = tf.get_variable('reset_weights', [int(np.prod(self.kernel_size))
                                                       * num_channels
                                                       * self.depth*depth_f],
                                     trainable=self.trainable)
                gr = tf.get_variable('reset_length', [],
                                     initializer=tf.random_normal_initializer(
                                         .0, 0.0001),
                                     trainable=self.trainable)
                vu = tf.get_variable('update_weights', (int(np.prod(self.kernel_size))
                                                        * self.depth
                                                        * self.depth*depth_f),
                                     trainable=self.trainable)
                gu = tf.get_variable('update_length', [],
                                     initializer=tf.random_normal_initializer(
                                         .0, 0.0001),
                                     trainable=self.trainable)
                # gr = tf.constant(0.0001)
                vb = tf.get_variable('bias', [self.depth*depth_f],
                                     trainable=self.trainable)
                gb = tf.get_variable('bias_length', [],
                                     initializer=tf.zeros_initializer(),
                                     trainable=self.trainable)

                if self.constrained:
                    gr = tf.nn.sigmoid(gr)
                    gu = tf.nn.sigmoid(gu)

                Wr = gr*tf.norm(vr)*tf.reshape(vr, (self.kernel_size
                                                    + [num_channels]
                                                    + [self.depth*depth_f]))
                Wu = gu*tf.norm(vu)*tf.reshape(vu, (self.kernel_size
                                                    + [self.depth]
                                                    + [self.depth*depth_f]))
                b = gb*tf.norm(vb)*vb
            else:
                Wr = tf.get_variable('reset_weights', (self.kernel_size
                                                       + [num_channels]
                                                       + [self.depth*depth_f]),
                                     trainable=self.trainable)
                Wu = tf.get_variable('update_weights', (self.kernel_size
                                                        + [self.depth]
                                                        + [self.depth*depth_f]),
                                     trainable=self.trainable)
                b = tf.get_variable('bias', [self.depth*depth_f],
                                    trainable=self.trainable,
                                    initializer=tf.zeros_initializer())

            # Seperate convolutions for inputs and recurrecies.
            # Slower but allows input downsampling.
            if self.transpose is False:
                input_conv = tf.nn.convolution(
                    x, Wr, 'SAME', strides=self.strides)
            else:
                input_conv = tf.nn.convolution(x, Wr, 'SAME')
            rec_conv = tf.nn.convolution(h, Wu, 'SAME')
            linear = input_conv + b
            z, r, h = tf.split(linear, 3, axis=self.recurrent_size.ndims)
            rz, rr, rh = tf.split(rec_conv, 3,
                                  axis=self.recurrent_size.ndims)
            z += rz
            r += rr
            z = tf.nn.sigmoid(z)
            r = tf.nn.sigmoid(r)
            h += r * rh
            h = self.activation(h)
            h = (1 - z) * state + z * h

            if self.output_depth is not None or self.transpose is True:
                with tf.variable_scope('output_proj'):
                    if self.transpose:
                        if self.output_depth is not None:
                            Wdeconv = tf.get_variable('input_weights', (self.kernel_size
                                                                        + [self.output_depth]
                                                                        + [self.depth]),
                                                      trainable=self.trainable)
                        else:
                            Wdeconv = tf.get_variable('input_weights', (self.kernel_size
                                                                        + [self.depth]
                                                                        + [self.depth]),
                                                      trainable=self.trainable)
                        shape = [batch_size, int(self.output_size[0]),
                                 int(self.output_size[1]), int(self.output_size[2])]
                        """
                        out = tf.nn.tanh(tf.nn.conv2d_transpose(
                            h, Wdeconv, strides=[1] + self.strides + [1],
                            output_shape=shape))
                        """
                        out = tf.nn.conv2d_transpose(
                            h, Wdeconv, strides=[1] + self.strides + [1],
                            output_shape=shape)
                    else:
                        Wproj = tf.get_variable('projection_weights',
                                                (self.kernel_size
                                                 + [self.depth]
                                                 + [self.output_depth]),
                                                trainable=self.trainable)
                        out = tf.nn.convolution(h, Wproj, padding='SAME')

                        shape = [batch_size, int(self.output_size[0]),
                                 int(self.output_size[1]), int(self.output_size[2])]
                        # out = tf.reshape(out, [batch_size] + self.input_dims[:2] + [-1])
                        out = tf.reshape(out, shape)
                        tf.Tensor.set_shape(out, shape)
            else:
                out = h
            if self.reuse is None:
                self.reuse = True
            return out, h


class TrajGRU(RNNCell):
    """implementation of trajectory GRU.
    reference: https://arxiv.org/abs/1706.03458
    """

    def __init__(self, links_num, kernel_size, depth, input_dims,
                 output_depth=None, strides=None, reuse=None,
                 activation=tf.nn.tanh):
        """ initialize trajectory GRU.
        links_num: the total number of links
        kernel_size: the size of the kernels
        depth: the kernel depth, or the number of filters
        input_dims: [height, width] channel is of no interest
        output_depth: the depth of the output convolution
        strides: the step sizes with which the sliding is done
        activation: activation function of hidden state and output
        """

        self._links_num = links_num
        self._kernel_size = kernel_size
        self._depth = depth
        self._input_dims = input_dims
        self._output_depth = output_depth
        if strides is None:
            self._strides = [1, 1]
        else:
            self._strides = strides
        self._reuse = reuse
        self._activation = activation

    @property
    def recurrent_size(self):
        """ shape of tensors flowing along the recurrent connections.
        """
        # the result with "same" padding
        return tf.TensorShape([np.ceil(self._input_dims[0] /
                                       self._strides[0]),
                               np.ceil(self._input_dims[1] /
                                       self._strides[1]), self._depth])

    @property
    def output_size(self):
        """ size of outputs produced by the cell.
        """
        if self._output_depth is None:
            return self.recurrent_size
        return tf.TensorShape([self.recurrent_size[0], self.recurrent_size[1],
                               self._output_depth])

    @property
    def state_size(self):
        """ size of the cell state.
        """
        return self.recurrent_size

    def _uv_generator(self, inputs, state, reuse, scope=None):
        """generate the local connection structure, stored in flow field.

        Args:
            inputs: current input
            state: previous hidden state

        Returns:
            u, v: each is tensor of shape (batch_size, h, w, self._links_num)
        """

        with tf.variable_scope('uv_generator'):
            combination = tf.concat([inputs, state], axis=-1)
            conv1 = tf.layers.conv2d(combination, 32, 5, padding='same',
                                     activation=tf.nn.leaky_relu,
                                     reuse=reuse, name='conv1')
            conv2 = tf.layers.conv2d(conv1, self._links_num*2, 5,
                                     padding='same', reuse=reuse,
                                     name='conv2')
            u, v = tf.split(conv2, 2, axis=-1)
            return u, v

    def __call__(self, inputs, state, scope=None):
        """
            Args:
                inputs: input of shape [batch_size, height, width, channels]
            Returns:
                outputs: with shape [batch_size] + self.output_size
                state: with shape [batch_size] + self.state_size
        """
        with tf.variable_scope(scope or str(type(self).__name__),
                               reuse=self._reuse):
            channels = inputs.shape[-1].value
            Wi = tf.get_variable('input_weights', (self._kernel_size
                                                   + [channels]
                                                   + [self._depth * 3]))
            Wr = tf.get_variable('recurrent_weights', ([1, 1]
                                                       + [self._depth *
                                                           self._links_num]
                                                       + [self._depth * 3]))
            b = tf.get_variable('state_bias', ([self.recurrent_size[0],
                                                self.recurrent_size[1]]
                                               + [self._depth * 3]))
            input_conv = tf.nn.convolution(inputs, Wi, 'SAME')
            linear = input_conv + b
            u, v = self._uv_generator(inputs, state, self._reuse)
            warped_state = simple_warp(state, u, v)
            recurrent_conv = tf.nn.convolution(warped_state, Wr, 'SAME')
            # z the update gate, r the reset gate, h the candidate state
            z, r, h = tf.split(linear, 3, axis=-1)
            rz, rr, rh = tf.split(recurrent_conv, 3,
                                  axis=-1)
            z += rz
            r += rr
            z = tf.nn.sigmoid(z)
            r = tf.nn.sigmoid(r)
            h += r * rh
            h = self._activation(h)
            h = (1 - z) * state + z * h

            if self._output_depth is not None:
                Wproj = tf.get_variable('projection_weights',
                                        (self._kernel_size + [self._depth]
                                         + [self._output_depth]))
                out = tf.nn.convolution(h, Wproj, 'SAME')
            else:
                out = h

            if self._reuse is None:
                self._reuse = True

            return out, h


class FlowGRU(RNNCell):
    """implementation of FlowGRU.
    """

    def __init__(self, kernel_size, depth, input_dims,
                 output_depth=None, strides=None, reuse=None,
                 activation=tf.nn.tanh):
        """ initialize FlowGRU.
        kernel_size: the size of the kernels
        depth: the kernel depth, or the number of filters
        input_dims: [height, width] channel is of no interest
        output_depth: the depth of the output convolution
        strides: the step sizes with which the sliding is done
        activation:
        """

        self._kernel_size = kernel_size
        self._depth = depth
        self._input_dims = input_dims
        self._output_depth = output_depth
        if strides is None:
            self._strides = [1, 1]
        else:
            self._strides = strides
        self._reuse = reuse
        self._activation = activation

    @property
    def recurrent_size(self):
        """ shape of tensors flowing along the recurrent connections.
        """
        # the result with "same" padding
        return tf.TensorShape([np.ceil(self._input_dims[0] /
                                       self._strides[0]),
                               np.ceil(self._input_dims[1] /
                                       self._strides[1]), self._depth])

    @property
    def output_size(self):
        """ size of outputs produced by the cell.
        """
        if self._output_depth is None:
            return self.recurrent_size
        return tf.TensorShape([self.recurrent_size[0], self.recurrent_size[1],
                               self._output_depth])

    @property
    def state_size(self):
        """ size of the cell state.
        """
        return self.recurrent_size

    def _uv_generator(self, inputs, state, reuse, scope=None):
        """generate the local connection structure, stored in flow field.

        Args:
            inputs:
            state:
            reuse:
            scope:

        Returns:
            u, v:each is tensor of shape (batch_size, h, w, 1)
        """

        with tf.variable_scope('uv_generator'):
            combination = tf.concat([inputs, state], axis=-1)
            filters = state.get_shape().as_list()[-1]
            conv1 = tf.layers.conv2d(combination, filters, 3, 2,
                                     padding='same',
                                     activation=tf.nn.leaky_relu,
                                     reuse=reuse, name='conv1')
            conv2 = tf.layers.conv2d(conv1, filters*2, 3, 2,
                                     padding='same',
                                     activation=tf.nn.leaky_relu,
                                     reuse=reuse, name='conv2')
            deconv1 = tf.layers.conv2d_transpose(conv2, filters, 3,
                                                 2, padding='same',
                                                 reuse=reuse, name='deconv1')
            deconv2 = tf.layers.conv2d_transpose(deconv1, int(filters/2), 3,
                                                 2, padding='same',
                                                 reuse=reuse, name='deconv2')
            conv3 = tf.layers.conv2d(deconv2, 2, 1,
                                     padding='same', reuse=reuse,
                                     name='conv3')
            u, v = tf.split(conv3, 2, axis=-1)
            return u, v

    def __call__(self, inputs, state, scope=None):
        """
            Args:
                inputs: input of shape [batch_size, height, width, channels]
            Returns:
                outputs: with shape [batch_size] + self.output_size
                state: with shape [batch_size] + self.state_size
        """
        with tf.variable_scope(scope or str(type(self).__name__),
                               reuse=self._reuse):
            channels = inputs.shape[-1].value
            Wi = tf.get_variable('input_weights', (self._kernel_size
                                                   + [channels]
                                                   + [self._depth * 3]))
            Wr = tf.get_variable('recurrent_weights', (self._kernel_size
                                                       + [self._depth]
                                                       + [self._depth * 3]))
            b = tf.get_variable('state_bias', ([self.recurrent_size[0],
                                                self.recurrent_size[1]]
                                               + [self._depth * 3]))
            input_conv = tf.nn.convolution(inputs, Wi, 'SAME')
            linear = input_conv + b
            u, v = self._uv_generator(inputs, state, self._reuse)
            warped_state = simple_warp(state, u, v)
            # pdb.set_trace()
            recurrent_conv = tf.nn.convolution(warped_state, Wr, 'SAME')
            # z the update gate, r the reset gate, h the candidate state
            z, r, h = tf.split(linear, 3, axis=-1)
            rz, rr, rh = tf.split(recurrent_conv, 3,
                                  axis=-1)
            z += rz
            r += rr
            z = tf.nn.sigmoid(z)
            r = tf.nn.sigmoid(r)
            h += r * rh
            h = self._activation(h)
            h = (1 - z) * state + z * h

            if self._output_depth is not None:
                Wproj = tf.get_variable('projection_weights',
                                        (self._kernel_size + [self._depth]
                                         + [self._output_depth]))
                out = tf.nn.convolution(h, Wproj, 'SAME')
            else:
                out = h

            if self._reuse is None:
                self._reuse = True

            return out, h


class FourierGRU(RNNCell):
    """Implement image transformation in trajectory GRU in frequency domain.
       reference: https://arxiv.org/abs/1706.03458
    """

    def __init__(self, objects_num, kernel_size, depth, input_dims,
                 output_depth=None, strides=None, reuse=None,
                 activation=tf.nn.tanh):
        """ initialize Fourier GRU.
        objects_num: the total number of objects
        kernel_size: the size of the kernels
        depth: the kernel depth, or the number of filters
        input_dims: [height, width] channel is of no interest
        output_depth: the depth of the output convolution
        strides: the step sizes with which the sliding is done
        activation:
        """

        self._objects_num = objects_num
        self._kernel_size = kernel_size
        self._depth = depth
        self._input_dims = input_dims
        self._output_depth = output_depth
        if strides is None:
            self._strides = [1, 1]
        else:
            self._strides = strides
        self._reuse = reuse
        self._activation = activation

    @property
    def recurrent_size(self):
        """ shape of tensors flowing along the recurrent connections.
        """
        # the result with "same" padding
        return tf.TensorShape([np.ceil(self._input_dims[0] /
                                       self._strides[0]),
                               np.ceil(self._input_dims[1] /
                                       self._strides[1]), self._depth])

    @property
    def output_size(self):
        """ size of outputs produced by the cell.
        """
        if self._output_depth is None:
            return self.recurrent_size
        return tf.TensorShape([self.recurrent_size[0], self.recurrent_size[1],
                               self._output_depth])

    @property
    def state_size(self):
        """ size of the cell state.
        """
        return self.recurrent_size

    def _affine_params_generator(self, inputs, state,
                                 reuse, arch='cnn', scope=None):
        """generate affine parameters, (vx, vy, theta).
            not considering scaling now.

        Args:
            inputs:
            state:
            arch: cnn or fcn
            reuse:
            scope:

        Returns:
            params: tensor of shape (batch_size, 3, self._objects_num)
        """

        with tf.variable_scope(scope or 'affine_params_generator'):
            '''
            if arch == 'cnn':
                fltrs=(16, 8)
                ks = (5, 3)
                combination = tf.concat([inputs, state], axis=-1)
                conv1 = tf.layers.conv2d(combination, fltrs[0], 
                                         ks[0], padding='same', 
                                         activation=tf.nn.leaky_relu,
                                         reuse=reuse, name='conv1')
                pool1 = tf.layers.max_pooling2d(conv1, [2, 2], 2)
                conv2 = tf.layers.conv2d(pool1, fltrs[1],
                                         ks[1], padding='same',
                                         activation=tf.nn.leaky_relu,
                                         reuse=reuse, name='conv2')
                pool2 = tf.layers.max_pooling2d(conv2, [2, 2], 2)
                pool2_flat = tf.layers.flatten(pool2, name='pool2_flat')
                dense1 = tf.layers.dense(pool2_flat, units=128,
                                         activation=tf.nn.relu, reuse=reuse)
                # initialized to have identity transform
                dense2 = tf.layers.dense(dense1, units=3*self._objects_num,
                                         kernel_initializer= \
                                                 tf.zeros_initializer(),
                                         bias_initializer= \
                                                 tf.zeros_initializer(),
                                         activation=tf.nn.tanh, reuse=reuse)
            return tf.reshape(dense2, (-1, 3, self._objects_num))
            '''
            # smaller network
            if arch == 'cnn':
                fltrs = (8, 4)
                ks = (5, 3)
                num_masks = self._objects_num
                combination = tf.concat([inputs, state], axis=-1)
                shape = state.get_shape().as_list()
                conv1 = tf.layers.conv2d(combination, fltrs[0],
                                         ks[0], (2, 2), padding='same',
                                         activation=tf.nn.leaky_relu,
                                         reuse=reuse, name='conv1')
                conv2 = tf.layers.conv2d(conv1, fltrs[1],
                                         ks[1], (2, 2), padding='same',
                                         activation=tf.nn.leaky_relu,
                                         reuse=reuse, name='conv2')
                conv2_flat = tf.layers.flatten(conv2, name='conv2_flat')
                dense = tf.layers.dense(conv2_flat, units=3*self._objects_num,
                                        kernel_initializer=tf.zeros_initializer(),
                                        bias_initializer=tf.zeros_initializer(),
                                        activation=tf.nn.tanh, reuse=reuse)
                deconv1 = tf.layers.conv2d_transpose(conv2, 8, 3, reuse=reuse,
                                                     strides=(2, 2),
                                                     padding='same', name='deconv1')
                deconv2 = tf.layers.conv2d_transpose(deconv1, 16, 5, reuse=reuse,
                                                     strides=(2, 2),
                                                     padding='same', name='deconv2')
                conv3 = tf.layers.conv2d(deconv2, num_masks+1, 1, padding='same',
                                         reuse=reuse, name='conv3')
                masks = tf.reshape(
                    tf.nn.softmax(tf.reshape(conv3, [-1, num_masks + 1])),
                    [shape[0], shape[1], shape[2], num_masks + 1])

            return tf.reshape(dense, (-1, 3, self._objects_num)), masks

    def __call__(self, inputs, state, scope=None):
        """
            Args:
                inputs: input of shape [batch_size, height, width, channels]
            Returns:
                outputs: with shape [batch_size] + self.output_size
                state: with shape [batch_size] + self.state_size
        """
        with tf.variable_scope(scope or str(type(self).__name__),
                               reuse=self._reuse):
            channels = inputs.shape[-1].value
            #state_channels = state.shape[-1].value
            Wi = tf.get_variable('input_weights', (self._kernel_size
                                                   + [channels]
                                                   + [self._depth * 3]))
            # Wr = tf.get_variable('recurrent_weights', ([1, 1]
            Wr = tf.get_variable('recurrent_weights', (self._kernel_size
                                                       + [self._depth]
                                                       + [self._depth * 3]))
            b = tf.get_variable('state_bias', ([self.recurrent_size[0],
                                                self.recurrent_size[1]]
                                               + [self._depth * 3]))
            input_conv = tf.nn.convolution(inputs, Wi, 'SAME')
            linear = input_conv + b
            affine_params, masks = self._affine_params_generator(inputs, state,
                                                                 self._reuse)
            warped_state = simple_fft_warp(state, affine_params, masks)
            recurrent_conv = tf.nn.convolution(warped_state, Wr, 'SAME')
            # z the update gate, r the reset gate, h the candidate state
            z, r, h = tf.split(linear, 3, axis=-1)
            rz, rr, rh = tf.split(recurrent_conv, 3,
                                  axis=-1)
            z += rz
            r += rr
            z = tf.nn.sigmoid(z)
            r = tf.nn.sigmoid(r)
            h += r * rh
            h = self._activation(h)
            h = (1 - z) * state + z * h

            if self._output_depth is not None:
                Wproj = tf.get_variable('projection_weights',
                                        (self._kernel_size + [self._depth]
                                         + [self._output_depth]))
                out = tf.nn.convolution(h, Wproj, 'SAME')
            else:
                out = h

            if self._reuse is None:
                self._reuse = True

            return out, h
