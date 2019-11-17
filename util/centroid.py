import torch
import numpy as np
import matplotlib.pyplot as plt
from moving_mnist_pp.movingmnist_iterator import MovingMNISTAdvancedIterator


def compute_2d_centroid(image):
    image_norm = image/torch.sum(image)
    x_max, y_max = image.shape[-2:]
    x = torch.arange(0, x_max)
    y = torch.arange(0, y_max)
    gridX, gridY = torch.meshgrid(x, y)
    gridX = gridX.unsqueeze(0).type(torch.float32)
    gridY = gridY.unsqueeze(0).type(torch.float32)
    weight_X = gridX*image_norm
    weight_Y = gridY*image_norm
    mean_x = torch.sum(weight_X.flatten(start_dim=1), dim=-1)
    mean_y = torch.sum(weight_Y.flatten(start_dim=1), dim=-1)
    return mean_x, mean_y


if __name__ == '__main__':
    it = MovingMNISTAdvancedIterator()
    time = 10
    seq_np, motion_vectors = it.sample(5, time)
    seq = torch.from_numpy(seq_np[:, :, 0, :, :].astype(np.float32))
    seq = seq[:, 0, :, :].unsqueeze(1)

    seq0 = seq[0, :, :, :]
    seq10 = seq[9, :, :, :]

    cent0 = compute_2d_centroid(seq0)
    print(cent0)
    cent10 = compute_2d_centroid(seq10)
    print(cent10)

    plt.imshow(seq0[0])
    plt.plot(cent0[1], cent0[0], 'r.')
    plt.show()
    plt.imshow(seq10[0])
    plt.plot(cent10[1], cent10[0], 'r.')
    plt.show()
