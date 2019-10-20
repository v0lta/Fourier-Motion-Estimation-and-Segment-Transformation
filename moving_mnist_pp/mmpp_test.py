import cv2
import matplotlib.pyplot as plt
from moving_mnist_pp.movingmnist_iterator import MovingMNISTAdvancedIterator

it = MovingMNISTAdvancedIterator()
seq, motion_vectors = it.sample(5, 20)

plt.imshow(seq[0, 0, 0, :, :])
plt.show()