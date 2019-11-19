import torch
import numpy as np
from util.rotation_translation_pytorch import complex_hadamard, fft_shift, complex_abs, complex_conj
import math

def register_translation(image1, image2):
    """
    Register the circular translation of image1 with respect to image2 in pixels.
    :param image1: Tensor of shape [batch_size, height, width]
    :param image2: Tensor of shape [batch_size, height, width]
    :return: vx, vy, gg_star
    """
    assert image1.shape == image2.shape, 'shape of both images must be identical.'
    batch_size = image1.shape[0]
    # compute the two dimensional fourier transforms.
    c_image1 = torch.stack([image1/255, torch.zeros_like(image1)], -1)
    c_image2 = torch.stack([image2/255, torch.zeros_like(image2)], -1)
    fft1 = torch.fft(c_image1, 2)
    fft2 = torch.fft(c_image2, 2)

    gg_star = complex_hadamard(fft1, complex_conj(fft2))
    small_r = torch.ifft(gg_star, 2)
    abs_small_r = complex_abs(small_r)
    abs_small_r = abs_small_r.reshape([batch_size, -1])
    max_index_flat = torch.argmax(abs_small_r, dim=-1)
    vx = max_index_flat / image1.shape[-2]
    vy = max_index_flat % image1.shape[-2]
    return vx, vy, gg_star


def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
                 else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
                 for i in range(x.dim()))
    return x[inds]

def log_polar(image, angles=None, radii=None):
    """Return log-polar transformed image and log base.
    TODO: Fixme using https://www.lfd.uci.edu/~gohlke/code/imreg.py.html
    TODO: Why is the output flipped? """
    print(image.shape)
    shape = image.shape[1:]
    center = shape[0] / 2, shape[1] / 2
    if angles is None:
        angles = shape[0]
    if radii is None:
        radii = shape[1]
    theta = torch.empty((angles, radii), dtype=torch.float32)
    theta.T[:] = torch.linspace(0,np.pi, angles) # * -1.0
    # d = radii
    x1 = shape[0] - center[0]
    x2 = shape[1] - center[1]
    d = np.sqrt(x1*x1 + x2*x2)
    log_base = torch.tensor(10.0 ** (np.log10(d) / radii))
    radius = torch.empty_like(theta)
    radius[:] = torch.pow(log_base,
                          torch.arange(radii).type(torch.float32)) - 1
    x = radius * (shape[0]/shape[1]) * torch.cos(theta) + center[0]
    y = radius * (shape[1]/shape[0]) * torch.sin(theta) + center[1]
    # print(x)
    # x = shape[0] - x
    y = shape[1] - y
    grid = torch.stack([(x/shape[0] - 0.5)*2, (y/shape[1] - 0.5)*2], dim=-1)
    output = torch.nn.functional.grid_sample(image.unsqueeze(0),
                                             grid.unsqueeze(0))
    # plt.imshow(output[0, 0, :, :].numpy()); plt.show()
    return output, log_base


def register_rotation(image1, image2):
    c_image1 = torch.stack([image1, torch.zeros_like(image1)], -1)
    c_image2 = torch.stack([image2, torch.zeros_like(image2)], -1)
    f0 = fft_shift(complex_abs(torch.fft(c_image1, 2)))
    f1 = fft_shift(complex_abs(torch.fft(c_image2, 2)))

    f0, base = log_polar(f0)
    f1, base = log_polar(f1)
    i0, i1, ir = register_translation(f0, f1)
    angle = 180.0 * i0.type(torch.float32) / ir.shape[0]
    scale = base ** i1.type(torch.float32)
    return angle, scale


if __name__ == '__main__':
    from scipy import misc
    import matplotlib.pyplot as plt
    import util.numpy_registration as npreg
    import util.rotation_translation_pytorch as tr

    face = misc.face()
    I = face
    # I = I[128:(512+128), 256:(512+256)]
    I = np.mean(I, axis=-1)

    rows, cols = I.shape
    I = np.pad(I, ((rows//2, rows//2), (cols//2, cols//2)))
    m, n = I.shape
    plt.imshow(I)
    plt.show()

    It = torch.tensor(I.astype(np.float32)).unsqueeze(0)
    # Itt = tr.fft_translation(It, torch.tensor(0.1), torch.tensor(0.15))
    Ittr = tr.fft_rotation(It, torch.tensor(0.1).unsqueeze(0))

    plt.imshow(Ittr[0, :, :].numpy())
    plt.show()

    It2 = torch.cat([It, It], 0)
    Ittr2 = torch.cat([Ittr, It], 0)
    vx, vy, ggstar = register_translation(It2, Ittr2)
    print(vx, vy)

    logpolarI, base_np, x, y = npreg.logpolar(I)
    logpolarIt, base_torch = log_polar(It)
    print('base diff', np.abs(base_np - base_torch.numpy()))
    print('logpolar diff', np.mean(np.abs(logpolarI - logpolarIt[0, 0, :, :].numpy())))
    plt.imshow(logpolarI)
    plt.show()

    plt.imshow(logpolarIt[0, 0, :, :])
    plt.show()

    plt.imshow(logpolarI - logpolarIt[0, 0, :, :].numpy())
    plt.show()

    angle, scale = register_rotation(It, Ittr)
    print(angle)
