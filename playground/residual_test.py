import cv2
import torch
import numpy as np
import skimage.feature
import matplotlib.pyplot as plt
from moving_mnist_pp.movingmnist_iterator import MovingMNISTAdvancedIterator
from cells.FourierGRU import fft_affine_trans, register_rotation, \
    register_translation, complex_hadamard, complex_conj, ifft_shift, exp_i_phi, \
    complex_abs


def complex_phase(omega):
    return torch.atan2(omega[..., 1], omega[..., 0])

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
vy = torch.tensor(4.)
theta = torch.tensor(0.)
translate = fft_affine_trans(imgt.unsqueeze(0).unsqueeze(0), vx, vy, theta)
plt.imshow(translate[0, 0, :, :])
plt.show()

res = skimage.feature.register_translation(img, img2)

vx, vy, _ = register_translation(imgt, imgt2)
print('registration result', res, 'me', vx, vy)

# angle, scale = register_rotation(imgt, translate[0, 0, :, :])
# print(angle, scale)

c_image1 = torch.stack([imgt/255, torch.zeros_like(imgt)], -1)
c_image2 = torch.stack([imgt2/255, torch.zeros_like(imgt2)], -1)
fft1 = torch.fft(c_image1, 2)
fft2 = torch.fft(c_image2, 2)

gg_star = complex_hadamard(fft1, complex_conj(fft2))


t = torch.tensor(2.)
m_pad, n_pad = imgt.shape
bs = 1
channels = 1
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

plt.imshow(complex_abs(omega).numpy()[0, 0, :, :])
plt.imshow(complex_phase(omega).numpy()[0, 0, :, :])
plt.show()