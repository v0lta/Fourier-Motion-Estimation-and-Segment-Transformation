"""
Implementation of affine transformation using fft.
supprting batched data with multiple channels
"""

import numpy as np
import tensorflow as tf
#import pdb; pdb.set_trace()


# todo place code in module.
def tf_ifft_shift(x):
    """Tensorflow implementation of numpy's ifftshift

    Args:
        x: shape(n, )
    """
    shape = x.get_shape().as_list()
    n = shape[0]
    p2 = n-(n+1)//2
    indices = tf.concat([tf.range(p2, n), tf.range(p2)], axis=0)
    y = tf.gather(x, indices, axis=0)
    return y


def tf_outer(a, b):
    """ Tensorflow implementation of numpy's outer."""
    a_flat = tf.reshape(a, [-1])
    b_flat = tf.reshape(b, [-1])
    a_mul = tf.expand_dims(a_flat, axis=-1)
    b_mul = tf.expand_dims(b_flat, axis=0)
    return tf.multiply(a_mul, b_mul)

def fft_affine_trans(image, affine_params, padding, passes=2):
    """implement affine transformation using fft.

    Args:
        image: image or feature maps of shape (b, m, n, c)
        affine_params: tuple of 5 or 6 elements, parameters of affine
                       transformation, (vx, vy, theta, sx, sy)
        padding: padding size
        passes: number of passes the transformation is decomposed,
                default is 3
    Return:
        transformed image or feature maps, shape (b, m, n, c)
    """
    with tf.variable_scope('fft_affine_trans'):
        bs, M_init, N_init, _ = image.get_shape().as_list()
        pad_image = tf.pad(image, [[0, 0], [padding, padding],
                                   [padding, padding], [0, 0]])
        cimage = tf.complex(pad_image, tf.zeros_like(pad_image))
        # vx, vy, theta, sx, sy = affine_params
        vx = affine_params[:, 0]
        vy = affine_params[:, 1]
        theta = affine_params[:, 2]
        rx = tf.tan(theta/2)
        ry = -tf.sin(theta)
        cimage = tf.complex(pad_image, tf.zeros_like(pad_image))
        # transposed to (b, c, m, n)
        cimage = tf.transpose(cimage, [0, 3, 1, 2])
        # fft_img = affine_freq_trans(cimage, vx, rx, sx)
        fft_img = affine_freq_trans(cimage, vx, rx)
          
        cfft_img = tf.complex(fft_img, tf.zeros_like(fft_img))
        # transposed to (b, c, n, m)
        cfft_img = tf.transpose(cfft_img, [0, 1, 3, 2])    
        # fft2_img = affine_freq_trans(cfft_img, vy, ry, sy)
        fft2_img = affine_freq_trans(cfft_img, vy, ry)
        
        cfft2_img = tf.complex(fft2_img, tf.zeros_like(fft2_img))
        # transposed to (b, c, m, n)
        cfft2_img = tf.transpose(cfft2_img, [0, 1, 3, 2])
        fft3_img = affine_freq_trans(cfft2_img, tf.zeros((bs, )), rx)
        
        # transposed back to (b, m, n, c)
        fft3_img = tf.transpose(fft3_img, [0, 2, 3, 1])
        # crop padding area
        '''
        start_M = int(padding*sy)
        stop_M = int((M_init+padding)*sy)
        start_N = int(padding*sx)
        stop_N = int((N_init+padding)*sx)
        '''   
        start_M = int(padding)
        stop_M = int((M_init+padding))
        start_N = int(padding)
        stop_N = int((N_init+padding))
        fft3_crop = fft3_img[:, start_M:stop_M, start_N:stop_N, :]
        #import pdb; pdb.set_trace()
        # resize to original size
        fft3_resize = tf.image.resize_image_with_crop_or_pad(fft3_crop, M_init,
                                                             N_init)
        return fft3_resize
        
# def affine_freq_trans(cimage, v, r, s):
def affine_freq_trans(cimage, v, r):
    """implement affine transformation using fft.

    Args:
        cimage: complex image or feature maps of shape (b, c, m, n),
                first two are numbers of batch and channels
        v: translation, shape (b, )
        r: rotation, shape (b, )
        # s: scale, shape (b, )
    """
    with tf.variable_scope('affine_freq_trans'):
        bs, chnnls, M_pad, N_pad = tf.Tensor.get_shape(cimage).as_list()

        M = M_pad
        N = N_pad
        # N = int(N_pad*s)

        shift_omega = tf.ones([bs, chnnls, M, N], dtype=tf.complex64)
        # translation
        v = tf.reshape(v, (bs, 1, 1, 1))
        cScale = tf.complex(0.0, -2*np.pi*v/N)
        #vecM = tf.tile(tf.linspace(-N/2.0, N/2.0, N), [M])
        vecM = tf.linspace(-N/2.0, N/2.0, N)
        vecM = tf.reshape(vecM, (N, ))
        vecM = tf_ifft_shift(vecM)
        matM = tf.reshape(vecM, [1, 1, 1, N])
        matM = tf.tile(matM, [bs, chnnls, M, 1])
        #matM = tf.reshape(matM, [M, N])
        cMatM = tf.complex(matM, tf.zeros_like(matM))
        omegaM = tf.exp(cScale * cMatM)
        # the shift fixes the blurrieness problem.
        #shift_omega = tf.multiply(tf_ifft_shift(omegaM), shift_omega)
        shift_omega = tf.multiply(omegaM, shift_omega)
        
        # rotation
        # debug_here()
        kxRange = tf.range(-tf.floor(M/2.0), tf.ceil(M/2.0))
        kxRange.set_shape([M])
        MxRange = tf.range(-tf.floor(N/2.0), tf.ceil(N/2.0))
        MxRange.set_shape([N])
        MxRange = tf_ifft_shift(MxRange)
        # kxRange = tf_ifft_shift(kxRange)
        matM = tf_outer(kxRange, MxRange)
        # matM = tf_outer(kxRange, MxRange)
        matM = tf.reshape(matM, (1, 1, M, N))
        matM = tf.tile(matM, (bs, chnnls, 1, 1))
        r = tf.reshape(r, (bs, 1, 1, 1))
        cScale = tf.complex(0.0, tf.cast(-2.0*np.pi*r/N, tf.float32))
        cMat = tf.complex(matM, tf.zeros_like(matM))
        shift_omega = tf.multiply(tf.exp(cScale * cMat), shift_omega)

        fftimage = tf.fft(cimage)
        # fftimage = shift_omega * fftimage
        
        '''
        # scale
        if s > 1:
            to_pad = int(N_pad*s - N_pad)
            nyqst = int(np.ceil((N_pad + 1)/2))
            print(nyqst)
            # a(1:nyqst,:) ; zeros(ny-m,n) ; a(nyqst+1:m,:)
            fftimage = tf.concat([fftimage[:, :, :, :nyqst],
                                  tf.zeros([bs, chnnls, M, to_pad],
                                           tf.complex64),
                                  fftimage[:, :, :, nyqst:]],
                                 axis=-1)
        elif s < 1:
            remove_left = int(np.ceil((N_pad - N)/2))
            remove_right = int(np.floor((N_pad - N)/2))
            #use ceil, otherwise the size of fftimage and shift_omega will
            #be different by 1, but the result is of no meaning
            #remove_right = int(np.ceil((N_pad - N_pad*s)/2))
            nyqst = int(np.ceil((N_pad + 1)/2))
            fftimage = tf.concat([fftimage[:, :, :, :nyqst-remove_left],
                                  fftimage[:, :, :, nyqst+remove_right:]],
                                 axis=-1)
        '''
        return tf.real(tf.ifft(shift_omega * fftimage))
