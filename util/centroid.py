import torch
import numpy as np
import matplotlib.pyplot as plt
from moving_mnist_pp.movingmnist_iterator import MovingMNISTAdvancedIterator


def compute_2d_centroid(image):
    """
    Computed the 2d image centroid.
    :param image: image tensor [batch_size, height, width]
    :return: [batch_size, 2]
    """
    image_norm = image/torch.sum(image.flatten(start_dim=1), dim=-1).unsqueeze(-1).unsqueeze(-1)
    x_max, y_max = image.shape[-2:]
    x = torch.arange(0, x_max).cuda()
    y = torch.arange(0, y_max).cuda()
    gridX, gridY = torch.meshgrid(x, y)
    gridX = gridX.unsqueeze(0).type(torch.float32)
    gridY = gridY.unsqueeze(0).type(torch.float32)
    weight_X = gridX*image_norm
    weight_Y = gridY*image_norm
    mean_x = torch.sum(weight_X.flatten(start_dim=1), dim=-1)
    mean_y = torch.sum(weight_Y.flatten(start_dim=1), dim=-1)
    return torch.stack([mean_x, mean_y], dim=-1)


if __name__ == '__main__':
    from util.rotation_translation_pytorch import fft_rotation, fft_translation
    it = MovingMNISTAdvancedIterator()
    time = 10
    seq_np, motion_vectors = it.sample(5, time)
    seq = torch.from_numpy(seq_np[:, :, 0, :, :].astype(np.float32))
    seq = seq[:, 0, :, :].unsqueeze(1)

    seq0 = seq[0, :, :, :]
    seq5 = seq[5, :, :, :]
    seq10 = seq[9, :, :, :]

    img_stack = torch.cat([seq0, seq5, seq10], 0)

    cent = compute_2d_centroid(img_stack)
    print('seq0 centroid', cent[0])
    print('seq10m centroid', cent[2])

    plt.imshow(img_stack[0])
    plt.plot(cent[0][1], cent[0][0], 'r.')
    plt.show()
    plt.imshow(img_stack[2])
    plt.plot(cent[2][1], cent[2][0], 'r.')
    plt.show()

    rot_img_stack = fft_rotation(seq10, torch.tensor([0.1, 0.2, 0.3]))
    rot_rot_img_stack_cent = compute_2d_centroid(rot_img_stack)
    print('rotated centroid', rot_rot_img_stack_cent[-1])
    # plt.imshow(rot_seq0[0])
    # plt.plot(rot_cent10[1], rot_cent10[0], 'r.')
    plt.show()

    # displacement
    displacement = cent - rot_rot_img_stack_cent
    print('pixel displacement', displacement)
    displacement = displacement/64.
    print('displacement', displacement)

    trans_rot_img_stack = fft_translation(rot_img_stack, displacement[:, 1], displacement[:, 0])
    trans_rot_img_stack_cent = compute_2d_centroid(trans_rot_img_stack)
    print(trans_rot_img_stack_cent[-1])
    plt.imshow(trans_rot_img_stack[-1])
    plt.plot(trans_rot_img_stack_cent[-1][1], trans_rot_img_stack_cent[-1][0], 'r.')
    plt.show()
    print('error', cent[2] - trans_rot_img_stack_cent[2])


    plt.imshow(img_stack[1])
    plt.plot(cent[1][1], cent[1][0], 'r.')
    plt.show()
    plt.imshow(trans_rot_img_stack[1])
    plt.plot(trans_rot_img_stack_cent[1][1], trans_rot_img_stack_cent[1][0], 'r.')
    plt.show()


