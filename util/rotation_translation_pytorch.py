import torch
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


def fft_shift(x: torch.Tensor) -> torch.Tensor:
    '''
    https://github.com/numpy/numpy/blob/v1.17.0/numpy/fft/helper.py#L22-L76
    :param x:
    :return:
    '''
    axes = tuple(range(x.ndim))
    shift = [dim // 2 for dim in x.shape]
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
    # assert ci1.shape[-1] == 2, 'we require real and imaginary part.'
    # assert ci2.shape[-1] == 2, 'we require real and imaginary part.'
    x1 = ci1[..., 0]
    y1 = ci1[..., 1]
    x2 = ci2[..., 0]
    y2 = ci2[..., 1]

    # multiplication in polar form is slow and numerically unstable in the backward pass.
    # rx = x1*x2 - y1*y2
    # ry = x1*y2 + y1*x2

    r1 = torch.sqrt(x1*x1 + y1*y1)
    phi1 = torch.atan2(y1, x1)
    r2 = torch.sqrt(x2*x2 + y2*y2)
    phi2 = torch.atan2(y2, x2)
    r = r1*r2
    phi = phi1 + phi2
    rx = torch.cos(phi)*r
    ry = torch.sin(phi)*r
    return torch.stack([rx, ry], -1)


def get_coords(col_no: torch.tensor):
    coords = torch.linspace(-np.fix(col_no/2,), np.ceil(col_no/2)-1, steps=col_no)
    shift = fft_shift(coords)
    return shift


def fft_shear_matrix(row_no, col_no, angle):
    shift = get_coords(col_no)
    i_vec = torch.tensor(range(0, row_no))-np.floor(row_no/2.)
    c_vec = -2*np.pi*shift*angle/row_no
    shear_mat = exp_i_phi(outer(i_vec, c_vec))
    return shear_mat


def fft_translation_matrix(row_no, col_no, t):
    shift = get_coords(col_no)
    shift = shift.repeat(row_no, 1)
    c_vec = -2*np.pi*t
    translation_matrix = exp_i_phi(c_vec*shift)
    return translation_matrix


def torch_fft_ifft(image, phase_modification_matrix, transpose=False):
    c_image = torch.stack([image, torch.zeros_like(image)], -1)
    # phase_modification_matrix = phase_modification_matrix.unsqueeze(0)
    if transpose:
        c_image = c_image.transpose(1, 2)
    image_trans = torch.ifft(complex_hadamard(torch.fft(c_image, signal_ndim=1),
                                              phase_modification_matrix),
                             signal_ndim=1)
    if transpose:
        image_trans = image_trans.transpose(1, 2)
    image_trans = image_trans[..., 0]
    return image_trans


def fft_translation(image, vx, vy):
    """

    :param image: [batch_size, height, width]
    :param vx: [batch_size] %TODO
    :param vy: [batch_size] %TODO
    :return: Teh translated image
    """
    # batch, height, width
    _, row_no, col_no = image.shape
    phase_modification_x = fft_translation_matrix(row_no, col_no, vx)
    image_trans_x = torch_fft_ifft(image, phase_modification_x)
    phase_modification_y = fft_translation_matrix(col_no, row_no, vy)
    image_trans_xy = torch_fft_ifft(image_trans_x, phase_modification_y, transpose=True)
    image_trans_xy = image_trans_x
    return image_trans_xy


def fft_rotation(image, theta):
    # batch, height, width
    _, row_no_init, col_no_init = image.shape
    image = torch.nn.functional.pad(image, [col_no_init//2, col_no_init//2,
                                            row_no_init//2, row_no_init//2])
    _, row_no, col_no = image.shape
    theta = theta*np.pi
    a = torch.tan(theta/2)
    b = -torch.sin(theta)

    phase_modification_x = fft_shear_matrix(row_no, col_no, a)
    image_shear_x = torch_fft_ifft(image, phase_modification_x)

    phase_modification_xy = fft_shear_matrix(col_no, row_no, b)
    image_shear_xy = torch_fft_ifft(image_shear_x, phase_modification_xy, transpose=True)

    phase_modification_xyz = fft_shear_matrix(row_no, col_no, a)
    image_shear_xyz = torch_fft_ifft(image_shear_xy, phase_modification_xyz)
    # remove the padding
    image_shear_xyz = image_shear_xyz[:, row_no_init//2:-row_no_init//2, col_no_init//2:-col_no_init//2]
    return image_shear_xyz


if __name__ == '__main__':
    import util.rotation_translation as np_rot_trans

    coords = get_coords(10)
    print('error coords', np.mean(np.abs(coords.numpy() - np_rot_trans.get_coords(10))))

    shear_mat = fft_shear_matrix(10, 10, .3)
    shear_mat = shear_mat[:, :, 0].numpy() + 1j*shear_mat[:, :, 1].numpy()
    print('error shear', np.mean(np.abs(shear_mat - np_rot_trans.fft_shear_matrix(10, 10, .3))))

    trans_mat = fft_translation_matrix(10, 10, .1)
    trans_mat = trans_mat[:, :, 0].numpy() + 1j*trans_mat[:, :, 1].numpy()
    print('error translation matrix', np.mean(np.abs(trans_mat - np_rot_trans.fft_translation_matrix(10, 10, .1))))

    face = misc.face()
    I = face
    # I = I[128:(512+128), 256:(512+256)]
    I = np.mean(I, axis=-1)

    rows, cols = I.shape
    I = np.pad(I, ((rows//2, rows//2), (cols//2, cols//2)))
    m, n = I.shape
    plt.imshow(I)
    plt.show()

    I_tensor = torch.tensor(I).unsqueeze(0)
    I_tensor_trans = fft_translation(I_tensor, torch.tensor(0.1), torch.tensor(0.15))
    I_array_trans = np_rot_trans.fft_translation(I, 0.1, 0.15)
    plt.imshow(I_tensor_trans[0, :, :].numpy())
    plt.show()
    print('error translation', np.mean(np.abs(I_tensor_trans[:, :].numpy() - I_array_trans)))

    Itr = fft_rotation(I_tensor_trans, theta=torch.tensor(0.5))
    Iar = np_rot_trans.fft_rotation(I_array_trans, theta=0.5)
    plt.imshow(Itr[0, :, :].numpy())
    plt.show()
    print('error rotation', np.mean(np.abs(Itr[0, :, :].numpy() - Iar)))

    Itr = fft_rotation(Itr, theta=torch.tensor(0.5))
    plt.imshow(Itr[0, :, :].numpy())
    plt.show()

    Itr = fft_rotation(Itr, theta=torch.tensor(0.5))
    plt.imshow(Itr[0, :, :].numpy())
    plt.show()

    Itr = fft_rotation(Itr, theta=torch.tensor(0.5))
    plt.imshow(Itr[0, :, :].numpy())
    plt.show()

    Itr = fft_translation(Itr, vx=-.1, vy=-.15)
    plt.imshow(Itr[0, :, :].numpy())
    plt.show()

    print(np.mean(np.abs(I_tensor - Itr)[0, :, :].numpy()))
