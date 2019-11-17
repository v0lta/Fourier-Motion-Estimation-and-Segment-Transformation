import torch
import numpy as np
from util.rotation_translation_pytorch import fft_translation
from util.pytorch_registration import register_translation
from moving_mnist_pp.movingmnist_iterator import MovingMNISTAdvancedIterator
import matplotlib.pyplot as plt
from util.write_movie import VideoWriter


class RegistrationCell(torch.nn.Module):
    """
    A fourier Domain registration and prediction cell.
    """

    def __init__(self, state_size=100, net_weight_size_lst=None):
        super().__init__()
        if net_weight_size_lst is None:
            net_weight_size_lst = [150, 150, 150]
        self.state_size = state_size
        self.net_weight_size_lst = net_weight_size_lst + [state_size]

        self.state_net = []
        activation = torch.nn.Tanh()
        in_size = 4 + self.state_size
        for layer_no, net_weight_no in enumerate(self.net_weight_size_lst):
            layer = torch.nn.Linear(in_size, net_weight_no)
            self.state_net.append(layer)
            self.state_net.append(activation)
            in_size = net_weight_no

        self.state_net = torch.nn.Sequential(*self.state_net)
        self.parameter_projection = torch.nn.Linear(self.state_size, 4)

    def forward(self, img, state):
        w, h = img.shape[-2:]
        w = w*1.0
        h = h*1.0
        state_vec, prev_img = state
        vy, vx, _ = register_translation(img, prev_img)
        print(vx, vy)
        if vx.numpy()[0] > w/2.0:
            vx = vx - w
        if vy.numpy()[0] > h/2.0:
            vy = vy - h
        print(vx, vy)
        vx = vx/w
        vy = vy/h
        print(vx, vy)
        #net_in = torch.cat([vx, vy, state_vec])
        #new_state_vec = self.state_net(net_in)
        #param_out = self.parameter_projection(new_state_vec)
        #vvx, vvy, vrx, vry = param_out

        pred_img = fft_translation(img, vx, vy)
        new_state = (state_vec, img)
        return pred_img, new_state


if __name__ == '__main__':
    print('hallo')
    it = MovingMNISTAdvancedIterator()
    time = 10
    seq_np, motion_vectors = it.sample(5, time)
    seq = torch.from_numpy(seq_np[:, :, 0, :, :].astype(np.float32))
    seq = seq[:, 0, :, :].unsqueeze(1)
    cell = RegistrationCell()
    out_lst = []
    zero_state = (torch.zeros([100]), seq[0, :, :, :])
    img = seq[1, :, :, :]
    for t in range(time - 1):
        img, new_state = cell.forward(img, zero_state)
        plt.imshow(img[0, :, :].numpy()); plt.show()
        out_lst.append(img)

    video = torch.stack(out_lst)
    write = np.concatenate([video.numpy(), seq_np[1:, 0]], -1)
    write = np.abs(write)/np.max(np.abs(write))
    video_writer = VideoWriter(height=64, width=128)
    video_writer.write_video(write[:, 0, :, :], filename='net_out.mp4')
    plt.show()

