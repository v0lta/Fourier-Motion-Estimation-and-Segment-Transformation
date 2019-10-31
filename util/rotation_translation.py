import numpy as np
import numpy.matlib
from scipy import misc
import matplotlib.pyplot as plt


def get_coords(col_no):
    coords = np.linspace(-np.fix(col_no/2,), np.ceil(col_no/2)-1, num=col_no)
    shift = np.fft.fftshift(coords)
    return shift


def fft_shear_matrix(row_no, col_no, angle):
    shift = get_coords(col_no)
    i_vec = np.array(range(0, row_no))-np.floor(row_no/2.)
    c_vec = -2*np.pi*shift*angle/row_no
    shear_mat = np.exp(1j*np.outer(i_vec, c_vec))
    return shear_mat


def fft_translation_matrix(row_no, col_no, t):
    shift = get_coords(col_no)
    shift = np.matlib.repmat(shift, row_no, 1)
    c_vec = -2*np.pi*t
    translation_matrix = np.exp(1j*c_vec*shift)
    return translation_matrix


def fft_translation(image, vx, vy):
    row_no, col_no = image.shape
    if vx != 0:
        phase_modification_x = fft_translation_matrix(row_no, col_no, vx)
        image_trans_x = np.fft.ifft(np.fft.fft(image)*phase_modification_x)
        image_trans_x = np.real(image_trans_x)
    else:
        image_trans_x = image
    if vy != 0:
        phase_modification_y = fft_translation_matrix(col_no, row_no, vy)
        image_trans_xy = np.fft.ifft(np.fft.fft(image_trans_x.transpose())*phase_modification_y).transpose()
        image_trans_xy = np.real(image_trans_xy)
    else:
        image_trans_xy = image_trans_x
    return image_trans_xy


def fft_rotation(image, theta):
    row_no_init, col_no_init = image.shape
    image = np.pad(image, ((row_no_init//2, row_no_init//2), (col_no_init//2, col_no_init//2)))
    row_no, col_no = image.shape
    theta = theta*np.pi
    a = np.tan(theta/2)
    b = - np.sin(theta)

    phase_modification_x = fft_shear_matrix(row_no, col_no, a)
    image_shear_x = np.fft.ifft(np.fft.fft(image)*phase_modification_x)
    image_shear_x = np.real(image_shear_x)

    phase_modification_xy = fft_shear_matrix(col_no, row_no, b)
    image_shear_xy = np.fft.ifft(np.fft.fft(image_shear_x.transpose())*phase_modification_xy).transpose()
    image_shear_xy = np.real(image_shear_xy)

    phase_modification_xyz = fft_shear_matrix(row_no, col_no, a)
    image_shear_xyz = np.fft.ifft(np.fft.fft(image_shear_xy)*phase_modification_xyz)
    image_shear_xyz = np.real(image_shear_xyz)
    # remove the padding
    image_shear_xyz = image_shear_xyz[row_no_init//2:-row_no_init//2, col_no_init//2:-col_no_init//2]
    return image_shear_xyz


if __name__ == '__main__':
    import util.numpy_registration as npreg
    face = misc.face()
    mean_face = np.mean(face, axis=-1)

    row_no, col_no = mean_face.shape
    I = np.pad(mean_face, ((row_no//2, row_no//2), (col_no//2, col_no//2)))
    m, n = I.shape
    plt.imshow(I)
    plt.show()

    It = fft_translation(I, vx=.1, vy=0.1)
    plt.imshow(It)
    plt.show()

    Itr = fft_rotation(It, theta=0.5)
    plt.imshow(Itr)
    plt.show()

    Itr = fft_rotation(Itr, theta=0.5)
    plt.imshow(Itr)
    plt.show()

    Itr4 = fft_rotation(Itr, theta=0.5)
    plt.imshow(Itr4)
    plt.show()

    Itr = fft_rotation(Itr4, theta=0.5)
    plt.imshow(Itr)
    plt.show()

    Itr = fft_translation(Itr, vx=-.1, vy=-.1)
    plt.imshow(Itr)
    plt.show()

    print(np.mean(np.abs(I - Itr)))

    # undo a translation using the registration function.
    I_crop = mean_face[128:(512+128), 256:(512+256)]
    rows, cols = I_crop.shape
    I_crop = np.pad(I_crop, ((rows//2, rows//2), (cols//2, cols//2)))
    rows, cols = I_crop.shape

    Ict = fft_translation(I_crop, 0.1, 0.15)
    # Ict = I_crop
    Ictr = fft_rotation(Ict, theta=0.5)
    # Ictr = Ict
    _, _, a, t, t_raw = npreg.similarity(I_crop, Ictr)
    print('angle, translation', a, t)
    # transform to radians
    a = a/180
    I_ir = fft_rotation(Ictr, -a)
    # I_rest = Ictr
    _, _, a2, t2, t_raw2 = npreg.similarity(I_crop, I_ir)
    I_it = fft_translation(I_ir, t_raw2[1]/rows, t_raw2[0]/cols)

    print(a, a2, t, t2, t_raw, t_raw2)
    plt.imshow(I_crop)
    plt.show()
    plt.imshow(Ictr)
    plt.show()
    plt.imshow(I_ir)
    plt.show()
