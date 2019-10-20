import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from moving_mnist_pp.movingmnist_iterator import MovingMNISTAdvancedIterator


def fft_shift(x: torch.Tensor) -> torch.Tensor:
    '''
    https://github.com/numpy/numpy/blob/v1.17.0/numpy/fft/helper.py#L22-L76
    :param x:
    :return:
    '''
    axes = tuple(range(x.ndim))
    shift = [dim // 2 for dim in x.shape]
    return torch.roll(x, shift, axes)


def ifft_shift(x: torch.Tensor) -> torch.Tensor:
    """Torch implementation of numpy's ifftshift
    Args:
        x: shape(n, )
    """
    # shape = list(x.shape)
    # n = shape[0]
    # p2 = n-(n+1)//2
    # indices = torch.cat([torch.arange(p2, n), torch.arange(0, p2)], dim=0)
    # y = torch.gather(x, index=indices, dim=0)
    axes = tuple(range(x.ndim))
    shift = [-(dim // 2) for dim in x.shape]
    return torch.roll(x, shift, axes)


def outer(a, b):
    """ Torch implementation of numpy's outer."""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul*b_mul


def exp_i_phi(real_in) -> torch.Tensor:
    return torch.stack([torch.cos(real_in), torch.sin(real_in)], -1)


def complex_conj(ci):
    assert ci.shape[-1] == 2,  'we require real and imaginary part in the last dimension.'
    return torch.stack([ci[..., 0], -ci[..., 1]], -1)


def complex_abs(ci):
    assert ci.shape[-1] == 2,  'we require real and imaginary part in the last dimension.'
    return torch.sqrt(ci[..., 0]*ci[..., 0] + ci[..., 1]*ci[..., 1])


def complex_hadamard(ci1, ci2):
    assert ci1.shape[-1] == 2, 'we require real and imaginary part.'
    assert ci2.shape[-1] == 2, 'we require real and imaginary part.'
    x1 = ci1[..., 0]
    y1 = ci1[..., 1]
    r1 = torch.sqrt(x1*x1 + y1*y1)
    phi1 = torch.atan2(y1, x1)

    x2 = ci2[..., 0]
    y2 = ci2[..., 1]
    r2 = torch.sqrt(x2*x2 + y2*y2)
    phi2 = torch.atan2(y2, x2)

    r = r1*r2
    phi = phi1 + phi2

    x = torch.cos(phi)*r
    y = torch.sin(phi)*r
    return torch.stack([x, y], -1)


def freq_interp(image, t, phi=0):
    '''
    Translation by t, rotation by phi.

    :param image: The omage to be transformed
    :param t: the translation parameter
    :param phi: the rotation parameter
    :return: the transformed image.
    '''
    bs, channels, m_pad, n_pad = image.shape
    m = m_pad
    n = n_pad
    # N = int(N_pad*s)

    omega = torch.ones([bs, channels, m, n, 2], dtype=torch.float32)
    if t != 0:
        # translation
        t = t.reshape((bs, 1, 1, 1))
        vec_m = torch.linspace(-n/2.0, n/2.0, n)
        vec_m = torch.reshape(vec_m, (n, ))
        vec_m = ifft_shift(vec_m)
        mat_m = torch.reshape(vec_m, [1, 1, 1, n])
        mat_m = mat_m.repeat([bs, channels, m, 1])
        c_scale = torch.tensor([-2*np.pi*t/n])
        omega_t = exp_i_phi(c_scale * mat_m)
        omega = complex_hadamard(omega_t, omega)

    if phi != 0:
        # rotation
        # TODO: Breaks sometimes, fixme!
        kx_range = torch.arange(-np.floor(m/2.0), np.ceil(m/2.0))
        mx_range = torch.arange(-np.floor(n/2.0), np.ceil(n/2.0))
        mx_range = ifft_shift(mx_range)
        kx_range = ifft_shift(kx_range)
        mat_m = outer(kx_range, mx_range)
        mat_m = outer(kx_range, mx_range)
        mat_m = torch.reshape(mat_m, (1, 1, m, n))
        mat_m = mat_m.repeat((bs, channels, 1, 1))
        phi = phi.reshape(bs, 1, 1, 1)
        c_scale = torch.tensor([-2.0*np.pi*phi/n])
        omega_phi = exp_i_phi(c_scale * mat_m)
        omega = complex_hadamard(omega_phi, omega)

    # TODO: scaling.

    c_image = torch.stack([image, torch.zeros_like(image)], -1)
    fft_image = torch.fft(c_image, 1)
    return torch.ifft(complex_hadamard(omega, fft_image), 1)[..., 0]


def fft_affine_trans(image, vx, vy, theta, padding=60):
    """implement affine transformation using fft.

    Args:
        image: image or feature maps of shape (b, m, n, c)
        vx: translation in x
        vy: translation in y
        theta: rotation
        padding: padding size
    Return:
        transformed image or feature maps, shape (b, m, n, c)
    """
    bs, _, m_init, n_init = image.shape
    pad_image = torch.nn.functional.pad(image, [padding, padding, padding, padding])
    # cimage = torch.complex(pad_image, tf.zeros_like(pad_image))
    # pad_image = image

    rx = torch.tan(theta/2)
    ry = -torch.sin(theta)
    fft_img = freq_interp(pad_image, vx, rx)
    fft2_img = freq_interp(fft_img.permute([0, 1, 3, 2]), vy, ry)
    fft3_img = freq_interp(fft2_img.permute([0, 1, 3, 2]), torch.zeros((bs, )), rx)
    # crop padding area
    '''
    start_M = int(padding*sy)
    stop_M = int((M_init+padding)*sy)
    start_N = int(padding*sx)
    stop_N = int((N_init+padding)*sx)
    '''
    # resize to original size
    start_M = int(padding)
    stop_M = int((m_init+padding))
    start_N = int(padding)
    stop_N = int((n_init+padding))
    fft3_crop = fft3_img[:, :, start_M:stop_M, start_N:stop_N]
    return fft3_crop


def register_translation(image1, image2):
    # compute the two dimensional fourier transforms.
    c_image1 = torch.stack([image1/255, torch.zeros_like(image1)], -1)
    c_image2 = torch.stack([image2/255, torch.zeros_like(image2)], -1)
    fft1 = torch.fft(c_image1, 2)
    fft2 = torch.fft(c_image2, 2)

    gg_star = complex_hadamard(fft1, complex_conj(fft2))
    small_r = torch.ifft(gg_star, 2)
    max_index_flat = torch.argmax(complex_abs(small_r))
    vx = max_index_flat / image1.shape[-2]
    vy = max_index_flat % image1.shape[-2]
    return vx, vy, gg_star


def register_rotation(image1, image2):
    c_image1 = torch.stack([image1/255, torch.zeros_like(image1)], -1)
    c_image2 = torch.stack([image2/255, torch.zeros_like(image2)], -1)
    f0 = fft_shift(complex_abs(torch.fft(c_image1, 2)))
    f1 = fft_shift(complex_abs(torch.fft(c_image2, 2)))

    def log_polar(image, angles=None, radii=None):
        """Return log-polar transformed image and log base.
        TODO: Fixme using https://www.lfd.uci.edu/~gohlke/code/imreg.py.html
        """
        shape = image.shape
        center = shape[0] / 2, shape[1] / 2
        if angles is None:
            angles = shape[0]
        if radii is None:
            radii = shape[1]
        theta = torch.empty((angles, radii), dtype=torch.float32)
        theta.T[:] = torch.linspace(0, np.pi, angles) * -1.0
        # d = radii
        x1 = shape[0] - center[0]
        x2 = shape[1] - center[1]
        d = np.sqrt(x1*x1 + x2*x2)
        log_base = torch.tensor(10.0 ** (np.log10(d) / radii))
        radius = torch.empty_like(theta)
        radius[:] = torch.pow(log_base,
                              torch.arange(radii).type(torch.float32)) - 1.0
        x = radius * torch.sin(theta) + center[0]
        y = radius * torch.cos(theta) + center[1]
        output = torch.empty_like(x)
        # ndii.map_coordinates(image, [x, y], output=output)
        grid = torch.stack([x/shape[0], y/shape[1]], dim=-1)
        output = torch.nn.functional.grid_sample(image.unsqueeze(0).unsqueeze(0), grid.unsqueeze(0))
        return output, log_base

    f0, base = log_polar(f0)
    f1, base = log_polar(f1)
    i0, i1, ir = register_translation(f0, f1)
    angle = 180.0 * i0.type(torch.float32) / ir.shape[0]
    scale = base ** i1.type(torch.float32)
    return angle, scale


if __name__ == "__main__":
    import skimage.feature

    # test the freq interp Alogrithm
    it = MovingMNISTAdvancedIterator()
    seq, motion_vectors = it.sample(5, 20)
    img = seq[0, 0, 0, :, :]
    plt.imshow(img)
    plt.show()

    img2 = seq[1, 0, 0, :, :]
    plt.imshow(img2)
    plt.show()

    imgt = torch.tensor(img)
    imgt2 = torch.tensor(img2)
    # t = torch.tensor(0.0)
    # phi = torch.tensor(1.0)
    # translate = freq_interp(img.unsqueeze(0).unsqueeze(0), 0, phi=phi)
    # plt.imshow(translate[0, 0, :, :])
    # plt.show()

    vx = torch.tensor(0.)
    vy = torch.tensor(0.)
    theta = torch.tensor(np.pi/3)
    translate = fft_affine_trans(imgt.unsqueeze(0).unsqueeze(0), vx, vy, theta)
    plt.imshow(translate[0, 0, :, :])
    plt.show()

    res = skimage.feature.register_translation(img, img2)

    vx, vy, _ = register_translation(imgt, imgt2)
    print(res, 'me', vx, vy)

    angle, scale = register_rotation(imgt, translate[0, 0, :, :])
    print(angle, scale)