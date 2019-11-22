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
plt.imshow(I)
plt.show()

theta = np.pi/3.
a = np.tan(theta/2)
b = - np.sin(theta)


def fft_shear(image, row_no, col_no, angle):
    coords = np.linspace(-np.fix(col_no/2,), np.ceil(col_no/2)-1, num=col_no)
    shift = np.fft.fftshift(coords)
    i_vec = np.array(range(0, row_no))-np.floor(row_no/2.)
    c_vec = -2*np.pi*shift*angle/row_no
    shear_mat = np.exp(1j*np.outer(i_vec, c_vec))
    image_shear = np.fft.ifft(np.fft.fft(image)*shear_mat)
    image_shear = np.real(image_shear)
    return image_shear


Ix = fft_shear(I, m, n, a)
plt.imshow(Ix)
plt.show()

Iy = fft_shear(Ix.transpose(), n, m, b).transpose()
plt.imshow(Iy)
plt.show()

If = fft_shear(Iy, m, n, a)
plt.imshow(If)
plt.show()


# rotate back
teta = -np.pi/3.
a = np.tan(teta/2)
b = - np.sin(teta)

IIx = fft_shear(If, m, n, a)
plt.imshow(IIx)
plt.show()

IIy = fft_shear(IIx.transpose(), n, m, b).transpose()
plt.imshow(IIy)
plt.show()

IIf = fft_shear(IIy, m, n, a)
plt.imshow(IIf)
plt.show()

print('error', np.mean(np.abs(I.flatten() - IIf.flatten())))