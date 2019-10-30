import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
face = misc.face()
I = face
# I = I[128:(512+128), 256:(512+256)]
I = np.mean(I, axis=-1)


m, n = I.shape
I = np.pad(I, ((m//2, m//2), (n//2, n//2)))
m, n = I.shape
plt.imshow(I.astype(np.uint32))
plt.show()

teta = np.pi/3.
a = np.tan(teta/2.)
b = - np.sin(teta)

range_nx = np.linspace(-np.fix(m/2,), np.ceil(m/2)-1, num=m)
range_ny = np.linspace(-np.fix(n/2,), np.ceil(n/2)-1, num=n)

Nx = np.fft.ifftshift(range_nx)
Ny = np.fft.ifftshift(range_ny)

Ix = np.zeros([m, n])
for k in range(0, m):
    shear_vec = np.exp(-2*1j*np.pi*(k-np.floor(m/2.))*Ny*a/m)
    # shear_vec = np.expand_dims(shear_vec, 0)
    Ix[k, :] = np.fft.ifft(np.fft.fft(I[k, :])*shear_vec)
    Ix = np.real(Ix)
plt.imshow(Ix)
plt.show()

Iy = np.zeros([m, n])
for k in range(0, n):
    # python wants a minus here!
    shear_vec = np.exp(-2*1j*np.pi*(k-np.floor(n/2.))*Nx*b/n)
    Iy[:, k] = np.fft.ifft(np.fft.fft(Ix[:, k])*shear_vec)
    Iy = np.real(Iy)
plt.imshow(Iy)
plt.show()


If = np.zeros([m, n])
for k in range(0, m):
    shear_vec = np.exp(-2*1j*np.pi*(k-np.floor(m/2.))*Ny*a/m)
    If[k, :] = np.fft.ifft(np.fft.fft(Iy[k, :])*shear_vec)
    If = np.real(If)

plt.imshow(If)
plt.show()

