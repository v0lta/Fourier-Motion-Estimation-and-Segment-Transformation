import tensorflow as tf
import numpy as np
import pdb
import math
from .fft_affine_trans import fft_affine_trans
def combine_(inputs, compress_rate=2):
    """ combine tensor.

    Args:
        inputs: tensor of shape [max_time, batch_size, h, w, d]
        compress_rate:

    Returns:
        compressed tensor of shape [max_time/compress_rate, batch_size,
                                    h, w, d*compress_rate]
    """
    with tf.variable_scope("combine"):
        max_time = inputs.shape[0].value
        if max_time % compress_rate != 0:
            raise ValueError("the compress rate %d is not compatible with \
                             length %d" % (compress_rate, max_time))
        compressed = []
        for i in range(compress_rate):
            compressed.append(tf.gather(inputs, tf.range(i, max_time, delta=compress_rate)))
        compressed = tf.concat(compressed, -1)
        return compressed


def split_(inputs, compress_rate=2):
    """split inputs into compress_rate pieces.

    Args:
        inputs: tensor of shape [max_time, batch_size, h, w, d].
        compress_rate:

    Returns:
        splited tensor of shape [max_time*compress_rate, batch_size, h, w,
                                 d/compress_rate].
    """
    with tf.variable_scope("split"):
        splited = tf.split(inputs, compress_rate, -1)
        splited = tf.concat(splited, 0)
        max_time = inputs.shape[0].value
        length = max_time * compress_rate
        index = []
        for i in range(max_time):
            index.append(tf.range(i, length, delta=max_time))
        index = tf.concat(index, -1)
        splited = tf.gather(splited, index)
        return splited

'''
def split_(inputs, compress_rate):
    """split inputs into compress_rate pieces.

    Args:
        inputs: tensor of shape [max_time, batch_size, h, w, d].
        compress_rate:

    Returns:
        splited tensor of shape [max_time*compress_rate, batch_size, h, w,
                                 d/compress_rate].
    """
    with tf.variable_scope("split"):
        shape = inputs.get_shape().as_list()
        mt = shape[0]
        bs = shape[1]
        h = shape[2]
        w = shape[3]
        d = shape[4]

        cr = compress_rate
        mt_out = mt * cr
        d_out = int(d / cr)

        #m_idx should have form [0, ..., 0, 1, ..., 1,
        #                        mt -1, ..., mt - 1]    
        m_idx = np.arange(0, mt)
        m_idx = np.tile(m_idx, (cr,))
        m_idx = np.split(m_idx, cr, 0)
        m_idx = np.stack(m_idx, -1)
        m_idx = np.reshape(m_idx, (mt_out,))

        b_idx = np.arange(0, bs)
        h_idx = np.arange(0, h)
        w_idx = np.arange(0, w)
        d_idx = np.arange(0, d_out)

        indices = np.meshgrid(m_idx, b_idx, h_idx, w_idx, d_idx, indexing='ij')
        indices = np.stack(indices, axis=-1)
        for i in range(mt):
            for j in range(cr - 1):
                indices[i * cr + (j + 1), :, :, :, :, -1] += d_out * (j + 1)
        result = tf.gather_nd(inputs, indices)
        return result

def combine_(inputs, compress_rate):
    """ combine tensor.

    Args:
        inputs: tensor of shape [max_time, batch_size, h, w, d]
        compress_rate:

    Returns:
        compressed tensor of shape [max_time/compress_rate, batch_size,
                                    h, w, d*compress_rate]
    """
    with tf.variable_scope("combine"):
        shape = inputs.get_shape().as_list()
        mt = shape[0]
        bs = shape[1]
        h = shape[2]
        w = shape[3]
        d = shape[4]

        cr = compress_rate
        mt_out = int(mt / cr)
        d_out = d * cr

        m_idx = np.zeros((mt_out,), dtype=int)    
        b_idx = np.arange(0, bs)
        h_idx = np.arange(0, h)
        w_idx = np.arange(0, w)
        d_idx = np.arange(0, d)
        d_idx = np.tile(d_idx, (cr,))
        indices = np.meshgrid(m_idx, b_idx, h_idx, w_idx, d_idx, indexing='ij')
        indices = np.stack(indices, axis=-1)
        inc = 0
        for i in range(mt_out):
            for j in range(d_out):
                indices[i, :, :, :, j, 0] += inc
                if (j+1) % d == 0:
                    inc += 1           
        result = tf.gather_nd(inputs, indices)
        return result
'''

def downsample(inputs, kernel_size, stride=2):
    """Downsample the inputs using convolution.

    Args:
        inputs: tensor of shape (time, batch_size, height, width, depth)
        kernel_size: the kernel of convolution
        stride: the stride of convolution

    Returns:
            downsampled inputs
    """
    with tf.variable_scope("downsample", reuse=tf.AUTO_REUSE):
        in_shape = inputs.get_shape().as_list()
        time = in_shape[0]
        bs = in_shape[1]
        depth = in_shape[-1]
        inputs = tf.reshape(inputs, [-1] + in_shape[2:])
        weights = tf.get_variable("weights", kernel_size + [depth, depth])
        outputs = tf.nn.conv2d(inputs, weights,
                               strides=[1, stride, stride, 1],
                               padding='SAME')
        outputs_shape = outputs.get_shape().as_list()
        return tf.reshape(outputs, [time, bs] + outputs_shape[1:])

def upsample(inputs, kernel_size, stride=2, out_shape=None):
    """Upsample the inputs using transposed convolution.

    Args:
        inputs: tensor of shape (time, batch_size, height, width, depth)
        kernel_size: the kernel of transposed convolution
        stride: the strides of transposed convolution

    Returns:
            downsampled inputs
    """
    with tf.variable_scope("upsample", reuse=tf.AUTO_REUSE):
        in_shape = inputs.get_shape().as_list()
        time = in_shape[0]
        bs = in_shape[1]
        depth = in_shape[-1]
        inputs = tf.reshape(inputs, [-1] + in_shape[2:])
        weights = tf.get_variable("weights", kernel_size + [depth, depth])
        if out_shape is None:
            out_shape = inputs.get_shape().as_list()
            out_shape[1] *= 2
            out_shape[2] *= 2
        outputs = tf.nn.conv2d_transpose(inputs,
                                         weights, 
                                         out_shape,
                                         strides=[1, stride, stride, 1],
                                         padding='SAME')
        outputs_shape = outputs.get_shape().as_list()
        return tf.reshape(outputs, [time, bs] + outputs_shape[1:])

def preprocess_data(data):
    """Permute the axis to make the channel the last one and scale.

    Args:
        data: [time, batch_size, channel, height, width]

    Returns:
        processed data of shape [time, batch_size, height, width, channel]
    """
    data = np.transpose(data, [0, 1, 3, 4, 2])
    #scale
    #to normalize in range [0, 1]
    #data = 2 * data / 255 + (-1)
    data = data / 255
    return data

def get_pixel_value(img, x, y):
    """get the pixel value.
    
    Args:
        img: tensor of shape (bs, h, w, depth)
        x: shape (bs, h, w, links_num)
        y: shape (bs, h, w, links_num)
    Returns:
        output: tensor of shape (bs, h, w, depth*links_num)
    """
    """get the pixel value.
    
    Args:
        img: tensor of shape (bs, h, w, depth)
        x: shape (bs, h, w, links_num)
        y: shape (bs, h, w, links_num)
    Returns:
        output: tensor of shape (bs, h, w, depth*links_num)
    """

    with tf.variable_scope("get_pixel_value"):
        shape = img.get_shape().as_list()
        bs = shape[0]
        h = shape[1]
        w = shape[2]
        d = shape[3]

        links_num = x.get_shape().as_list()[-1]
        c = d * links_num

        b_idx = tf.range(0, bs)
        h_idx = tf.range(0, h)
        w_idx = tf.range(0, w)
        #l_idx should have form [0, ..., 0, 1, ..., 1, link_num-1, ...,
        #                        links_num-1]
        
        l_idx = tf.range(0, links_num)
        l_idx = tf.tile(l_idx, (d,))
        l_idx = tf.split(l_idx, d, 0)
        l_idx = tf.stack(l_idx, -1)
        l_idx = tf.reshape(l_idx, (c,))
    
        xy_idx = tf.meshgrid(b_idx, h_idx, w_idx, l_idx, indexing='ij')
        xy_idx = tf.stack(xy_idx, -1)
        
        x_idx = tf.gather_nd(x, xy_idx, name='x_idx')
        y_idx = tf.gather_nd(y, xy_idx, name='y_idx')

        #c_idx should have form [0, ..., d-1, 0, ..., d-1]    
        c_idx = tf.range(0, d)
        c_idx = tf.tile(c_idx, (links_num,))
        c_idx = tf.reshape(c_idx, (c,))
        b_idx, _, _, c_idx = tf.meshgrid(b_idx, h_idx, w_idx, c_idx,
                                         indexing='ij')

        indices = tf.stack([b_idx, x_idx, y_idx, c_idx], -1, name='indices')
        return tf.gather_nd(img, indices, name='warped_pixel')

def bi_interpolation(features, x, y):
    """bilinear interpolation in a tensor way
    Args:
        features: shape (batch_size, h, w , depth)
        x: shape (batch_size, h, w, links_num)
    """
    with tf.variable_scope('bi_interpolation'):
        shape = features.get_shape().as_list()
        bs = shape[0]
        h = shape[1]
        w = shape[2]
        d = shape[3]
        links_num = x.get_shape().as_list()[-1]
        c = links_num * d
        
        max_x = h -1
        max_y = w -1
        zero = 0
        #x = tf.clip_by_value(x, zero, max_x) 
        #y = tf.clip_by_value(y, zero, max_y) 
        reg_x0 = tf.floor(x)
        reg_x1 = reg_x0 + 1
        reg_y0 = tf.floor(y)
        reg_y1 = reg_y0 + 1

        reg_x0 = tf.clip_by_value(reg_x0, zero, max_x, name='reg_x0')
        reg_x1 = tf.clip_by_value(reg_x1, zero, max_x, name='reg_x1')
        reg_y0 = tf.clip_by_value(reg_y0, zero, max_y, name='reg_y0')
        reg_y1 = tf.clip_by_value(reg_y1, zero, max_y, name='reg_y1')

        dd = x - reg_x0
        dt = 1 - dd
        dl = y - reg_y0
        dr = 1 - dl

        wa = dr * dt
        wb = dr * dd
        wc = dl * dd
        wd = dl * dt
        
        b_idx = tf.range(0, bs)
        h_idx = tf.range(0, h)
        w_idx = tf.range(0, w)
        #l_idx should have form [0, ..., 0, 1, ..., 1, link_num-1, ...,
        #                        links_num-1]
        l_idx = tf.range(0, links_num)
        l_idx = tf.tile(l_idx, (d,))
        l_idx = tf.split(l_idx, d, 0)
        l_idx = tf.stack(l_idx, -1)
        l_idx = tf.reshape(l_idx, (c,))
        indices = tf.meshgrid(b_idx, h_idx, w_idx, l_idx, indexing='ij')
        indices = tf.stack(indices, -1, name='indices')
        
        wa = tf.gather_nd(wa, indices)
        wb = tf.gather_nd(wb, indices)
        wc = tf.gather_nd(wc, indices)
        wd = tf.gather_nd(wd, indices)
        
        reg_x0 = tf.cast(reg_x0, tf.int32)
        reg_x1 = tf.cast(reg_x1, tf.int32)
        reg_y0 = tf.cast(reg_y0, tf.int32)
        reg_y1 = tf.cast(reg_y1, tf.int32)

        Ia = get_pixel_value(features, reg_x0, reg_y0)
        Ib = get_pixel_value(features, reg_x1, reg_y0)
        Ic = get_pixel_value(features, reg_x1, reg_y1)
        Id = get_pixel_value(features, reg_x0, reg_y1)

        return tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id], name='warped_img')

def depthwise_warp(features, u, v):
    """warp the features according to the flow field depth by depth.

    Args:
        features: tensof of shape [batch_size, h, w, depth]
        u: x direction flow, of shape [batch_size, h, w, links_num]
        v: y direction flow, of shape [batch_size, h, w, links_num]
    Returns:
        warped features
    """
    with tf.variable_scope('warp'):
        shape = features.get_shape().as_list()
        bs = shape[0]
        h = shape[1]
        w = shape[2]
        #d = features.shape[3].value
        #links_num = u.shape[3].value

        x = tf.range(0, h)
        y = tf.range(0, w)
        u = u * h
        v = v * w
        xx, yy = tf.meshgrid(x, y, indexing='ij')

        xx = tf.reshape(xx, (1, h, w, 1))
        yy = tf.reshape(yy, (1, h, w, 1))
        # source_pos_x(or y): tensor of shape (batch_size, h, w, links_num)
        # require xx and u have the same dtype
        xx = tf.cast(xx, dtype=tf.float32)
        yy = tf.cast(yy, dtype=tf.float32)
        
        source_pos_x = tf.subtract(xx, u, name='source_pos_x')
        source_pos_y = tf.subtract(yy, v, name='source_pos_y')
        return bi_interpolation(features, source_pos_x, source_pos_y)
        
def my_ifftshift(x):
    """similar to numpy ifftshift.
    
    Args:
        x: a scalar
    returns:
    """
    low = tf.round(-x / 2)
    high =  tf.ceil(x / 2)
    low = tf.cast(low, tf.int32)
    high = tf.cast(high, tf.int32)
    x = tf.range(low, high)
    x = tf.split(x, [-low, high])
    x = tf.concat([x[1], x[0]], 0)
    return x
    
def warp_fft(features, u, v):
    """warp the features according to the flow field depth by depth.

    Args:
        features: tensof of shape [batch_size, h, w, depth]
        u: x direction flow, of shape [batch_size, h, w, links_num]
        v: y direction flow, of shape [batch_size, h, w, links_num]
    Returns:
        warped features
    """       
    bs, h, w, d = features.get_shape().as_list()
    _, _, _, links_num = u.get_shape().as_list()
    c = d * links_num
    u = u * w
    v = v * h
    b_idx = tf.range(0, bs)
    h_idx = tf.range(0, h)
    w_idx = tf.range(0, w)
    #l_idx should have form [0, ..., 0, 1, ..., 1, link_num-1, ...,
    #                        links_num-1]
        
    l_idx = tf.range(0, links_num)
    l_idx = tf.tile(l_idx, (d,))
    l_idx = tf.split(l_idx, d, 0)
    l_idx = tf.stack(l_idx, -1)
    l_idx = tf.reshape(l_idx, (c,))
    
    xy_idx = tf.meshgrid(b_idx, h_idx, w_idx, l_idx, indexing='ij')
    xy_idx = tf.stack(xy_idx, -1)
    u = tf.gather_nd(u, xy_idx)
    v = tf.gather_nd(v, xy_idx)
    
    features = tf.tile(features, (1, 1, 1, links_num))

    features = tf.transpose(features, [0, 3, 1, 2])
    u = tf.transpose(u, [0, 3, 1, 2])
    v = tf.transpose(v, [0, 3, 1, 2])

    Nr = my_ifftshift(w)
    Nc = my_ifftshift(h)
    Nr = tf.reshape(Nr, (1, 1, 1, w))
    Nc = tf.reshape(Nc, (1, 1, h, 1))

    Nr = tf.cast(Nr, tf.float32)
    Nc = tf.cast(Nc, tf.float32)
    #tf.multiply(x, y) x and y have to be the same type
    tmp = Nr*u/w + Nc*v/h 
    tmp = tf.cast(tmp, tf.complex64)
    translation_freq = tf.exp(-2*1j*math.pi*tmp)
    features = tf.cast(features, tf.complex64)
    shifted_freq = translation_freq * tf.fft2d(features)
    shifted_img = tf.ifft2d(shifted_freq)
    shifted_img = tf.real(shifted_img)
    shifted_img = tf.abs(shifted_img)
    shifted_img = tf.transpose(shifted_img, [0, 2, 3, 1])
    return shifted_img

def simple_warp(features, u, v):
    with tf.variable_scope('simple_warp'):
        bs, h, w, d = features.get_shape().as_list()
        links_num = u.get_shape().as_list()[-1]
        x = tf.range(0, h)
        y = tf.range(0, w)
        u = u * h
        v = v * w
        xx, yy = tf.meshgrid(x, y, indexing='ij')
        xx = tf.reshape(xx, (1, h, w, 1))
        yy = tf.reshape(yy, (1, h, w, 1))
        # source_pos_x(or y): tensor of shape (batch_size, h, w, links_num)
        # require xx and u have the same dtype
        xx = tf.cast(xx, dtype=tf.float32)
        yy = tf.cast(yy, dtype=tf.float32)
        source_pos_x = tf.subtract(xx, u, name='source_pos_x')
        source_pos_y = tf.subtract(yy, v, name='source_pos_y')

        warpped_features = tf.TensorArray(dtype=tf.float32, size=links_num)
        init_state = (0, warpped_features) 
        cond = lambda i, ta: tf.less(i, links_num)
        body = lambda i, ta: (i + 1,
                              ta.write(i, simple_biinterpolation(features,
                                                source_pos_x[: ,: ,: ,i],
                                                source_pos_y[:, :, :, i])))
        _, warpped_features = tf.while_loop(cond, body, init_state)
        warpped_features_stack = warpped_features.stack()
        warpped_features_stack.set_shape([links_num, bs, h, w, d]) 
        warpped_features_unstack = tf.unstack(warpped_features_stack,
                                              links_num)
        warpped_features_concat = tf.concat(warpped_features_unstack, -1) 
        return warpped_features_concat


def simple_biinterpolation(features, x, y):
    """bilinear interpolation in a tensor way.
    
    Args:
        features: shape (batch_size, h, w , depth)
        x, y: shape (batch_size, h, w)
    """
    with tf.variable_scope('simple_biinterpolation'):
        bs, h, w, d = features.get_shape().as_list()
        
        max_x = h -1
        max_y = w -1
        zero = 0
 
        reg_x0 = tf.floor(x)
        reg_x1 = reg_x0 + 1
        reg_y0 = tf.floor(y)
        reg_y1 = reg_y0 + 1

        reg_x0 = tf.clip_by_value(reg_x0, zero, max_x, name='reg_x0')
        reg_x1 = tf.clip_by_value(reg_x1, zero, max_x, name='reg_x1')
        reg_y0 = tf.clip_by_value(reg_y0, zero, max_y, name='reg_y0')
        reg_y1 = tf.clip_by_value(reg_y1, zero, max_y, name='reg_y1')

        dd = x - reg_x0
        dt = 1 - dd
        dl = y - reg_y0
        dr = 1 - dl

        wa = dr * dt
        wb = dr * dd
        wc = dl * dd
        wd = dl * dt
        
        wa = tf.expand_dims(wa, -1)
        wb = tf.expand_dims(wb, -1)
        wc = tf.expand_dims(wc, -1)
        wd = tf.expand_dims(wd, -1)
        
        reg_x0 = tf.cast(reg_x0, tf.int32)
        reg_x1 = tf.cast(reg_x1, tf.int32)
        reg_y0 = tf.cast(reg_y0, tf.int32)
        reg_y1 = tf.cast(reg_y1, tf.int32)

        Ia = simple_get_pixel_value(features, reg_x0, reg_y0)
        Ib = simple_get_pixel_value(features, reg_x1, reg_y0)
        Ic = simple_get_pixel_value(features, reg_x1, reg_y1)
        Id = simple_get_pixel_value(features, reg_x0, reg_y1)
        return tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id], name='warped_features')
        
def simple_get_pixel_value(img, x, y):
    """get the pixel value.
    
    Args:
        img: tensor of shape (bs, h, w, depth)
        x: shape (bs, h, w)
        y: shape (bs, h, w)
    Returns:
        output: tensor of shape (bs, h, w, depth)
    """         
    with tf.variable_scope("simple_get_pixel_value"):
        bs, h, w, d = img.get_shape().as_list()
        x = tf.expand_dims(x, -1)
        y = tf.expand_dims(y, -1)
        x = tf.tile(x, (1, 1, 1, d))
        y = tf.tile(y, (1, 1, 1, d))
         
        b_idx = tf.range(0, bs)
        h_idx = tf.range(0, h)
        w_idx = tf.range(0, w)
        d_idx = tf.range(0, d)
        b_idx, _, _, d_idx = tf.meshgrid(b_idx, h_idx, w_idx, d_idx,
                                         indexing='ij')
        indices = tf.stack([b_idx, x, y, d_idx], -1, name='indices')
        return tf.gather_nd(img, indices, name='warped_pixel')
        
def simple_fft_warp(features, params, masks):
    with tf.variable_scope('simple_fft_warp'):
        bs, h, w, d = features.get_shape().as_list()
        objects_num = params.get_shape().as_list()[-1]
        padding = int(h / 8)
        warpped_features = tf.TensorArray(dtype=tf.float32, size=objects_num) 
        init_state = (0, warpped_features)
        cond = lambda i, ta: tf.less(i, objects_num)
        body = lambda i, ta: (i + 1,
                              ta.write(i, fft_affine_trans(features,
                                                           params[:, :, i],
                                                           padding)))

        _, warpped_features = tf.while_loop(cond, body, init_state)
        mask_list = tf.split(axis=3, num_or_size_splits=objects_num+ 1,
                             value=masks)
        output = mask_list[0] * features
        for i in range(objects_num):
             output += warpped_features.read(i) * mask_list[i + 1]
        return output
