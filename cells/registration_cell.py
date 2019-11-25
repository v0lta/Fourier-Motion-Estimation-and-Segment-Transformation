import torch
import numpy as np
from util.rotation_translation_pytorch import fft_translation, fft_rotation
from util.pytorch_registration import register_translation, register_rotation
from moving_mnist_pp.movingmnist_iterator import MovingMNISTAdvancedIterator
from util.centroid import compute_2d_centroid
import matplotlib.pyplot as plt
from util.write_movie import VideoWriter, write_to_figure


class RegistrationCell(torch.nn.Module):
    """
    A fourier Domain fft-estimation, RNN-correction, fft-translation cell.
    """

    def __init__(self, state_size=100, net_weight_size_lst=None, learn_param_net=True,
                 gru=True, rotation=False):
        super().__init__()
        if net_weight_size_lst is None:
            assert gru is True
        else:
            self.net_weight_size_lst = net_weight_size_lst + [state_size]
        self.state_size = state_size

        self.state_net = []
        activation = torch.nn.Tanh()

        self.gru = gru
        self.rotation = rotation
        if self.rotation:
            self.in_size = 5
        else:
            self.in_size = 4

        if self.gru is True:
            in_size = self.in_size  # + self.state_size
            self.state_net = torch.nn.GRUCell(in_size, state_size)
        else:
            in_size = self.in_size + self.state_size # 4 + self.state_size
            for layer_no, net_weight_no in enumerate(self.net_weight_size_lst):
                layer = torch.nn.Linear(in_size, net_weight_no)
                self.state_net.append(layer)
                self.state_net.append(activation)
                in_size = net_weight_no
                self.state_net = torch.nn.Sequential(*self.state_net)
        self.parameter_projection = torch.nn.Linear(self.state_size, 3)
        self.learn_param_net = learn_param_net

    def forward(self, img, state):
        """
        :param img: [batch_size, 64, 64] image tensor
        :param state: ([batch_size, state
        :return: predicted image, new state
        """
        w, h = img.shape[-2:]
        w = w*1.0
        h = h*1.0
        state_vec, prev_img = state
        centroid = compute_2d_centroid(img)
        cx = centroid[:, 0]
        cy = centroid[:, 1]
        vy, vx, _ = register_translation(img, prev_img)
        # print(vx, vy)
        if vx.cpu().numpy()[0] > w/2.0:
            vx = vx - w
        if vy.cpu().numpy()[0] > h/2.0:
            vy = vy - h
        vx = vx/w
        vy = vy/h

        if self.rotation:
            rz = register_rotation(img, prev_img)[0]
            rz = rz*np.pi/180.
            estimation_vec = torch.cat([cx.unsqueeze(-1),
                                        cy.unsqueeze(-1),
                                        vx.unsqueeze(-1),
                                        vy.unsqueeze(-1),
                                        rz.unsqueeze(-1)], dim=-1)
        else:
            estimation_vec = torch.cat([cx.unsqueeze(-1),
                                        cy.unsqueeze(-1),
                                        vx.unsqueeze(-1),
                                        vy.unsqueeze(-1)], dim=-1)
            rz = 0

        if self.learn_param_net:
            if self.gru:
                new_state_vec = self.state_net(estimation_vec, state_vec)
                param_out = self.parameter_projection(new_state_vec)
            else:
                net_in = torch.cat([estimation_vec, state_vec], dim=-1)
                new_state_vec = self.state_net(net_in)
                param_out = self.parameter_projection(new_state_vec)

            vvx, vvy, vrz = torch.unbind(param_out, dim=-1)
            # vvx, vvy, _, _ = torch.unbind(param_out, dim=-1)
            vx += vvx
            vy += vvy
            rz += vrz
            # rz = 0
            state_vec = new_state_vec

        pred_img = fft_translation(img, vx, vy)

        if self.rotation:
            cent = compute_2d_centroid(pred_img)
            # move to the center
            displacement_x = (32 - cent[:, 1])/64.
            displacement_y = (32 - cent[:, 0])/64.

            cent_img = fft_translation(pred_img, displacement_x, displacement_y)
            # rotate
            rot_pred_img = fft_rotation(cent_img, rz)
            # move back
            rot_trans_pred_img = fft_translation(rot_pred_img, -displacement_x, -displacement_y)

            debug = False
            if debug:
                print(cent, displacement_x, displacement_y)
                print('cent', cent)
                plt.imshow(torch.cat([cent_img, rot_pred_img, rot_trans_pred_img], -1).cpu()[0, :, :])
                plt.show()

            pred_img = torch.clamp(rot_trans_pred_img, 0., 1.)
            # pred_img = rot_trans_pred_img

        new_state = (state_vec, img)
        return pred_img, new_state


class GatedRecurrentUnitWrapper(torch.nn.Module):
    """
    Wrap a GRU so that it works with the interface used here.
    """
    def __init__(self, state_size):
        super().__init__()
        self.cell = torch.nn.GRUCell(64*64, state_size)
        self.proj = torch.nn.Linear(state_size, 64*64)

    def forward(self, img, state):
        batch_size = img.shape[0]
        state_vec, prev_img = state
        flat_img = torch.reshape(img, [batch_size, -1])
        new_state_vec = self.cell(flat_img, state_vec)
        pred_vec = self.proj(new_state_vec)
        pred_img = torch.reshape(pred_vec, [batch_size, 64, 64])
        new_state = (new_state_vec, pred_img)
        return pred_img, new_state


class VelocityEstimationCell(torch.nn.Module):
    """
    A fourier Domain fft-prediction RNN correction cell.
    """
    def __init__(self, cnn_depth_lst, state_size=100, gru=True, phase_registration=False):
        super().__init__()
        self.cnn_depth_lst = cnn_depth_lst
        self.state_size = state_size

        cnn_lst = []
        activation = torch.nn.ReLU()
        in_depth = 2
        for cnn_depth in cnn_depth_lst:
            cnn_lst.append(torch.nn.Conv2d(in_depth, cnn_depth, kernel_size=3, padding=1))
            cnn_lst.append(activation)
            in_depth = cnn_depth
        self.cnn = torch.nn.Sequential(*cnn_lst)

        self.phase_registration = phase_registration
        if not self.phase_registration:
            self.state_net = []
            in_size = 2 + self.state_size  # 4 + self.state_size
            self.gru = gru
            if self.gru is True:
                self.state_net = torch.nn.GRUCell(self.cnn_depth_lst[-1]*64*64, state_size)
            self.rnn_parameter_projection = torch.nn.Linear(self.state_size, 4)
        else:
            self.state_net = []
            in_size = 4
            self.gru = gru
            if self.gru is True:
                self.state_net = torch.nn.GRUCell(in_size, state_size)
            self.rnn_parameter_projection = torch.nn.Linear(self.state_size, 4)
            self.cnn_parameter_projection = torch.nn.Linear(self.cnn_depth_lst[-1]*64*64, 4)

    def forward(self, img, state):
        """
        :param img: [batch_size, 64, 64] image tensor
        :param state: ([batch_size, state
        :return:
        """
        if not self.phase_registration:
            batch_size = img.shape[0]
            state_vec, prev_img = state
            # centroid = compute_2d_centroid(img)
            # cx = centroid[:, 0]
            # cy = centroid[:, 1]
            # vy, vx, _ = register_translation(img, prev_img)
            cnn_out = self.cnn(torch.stack([img, prev_img], -3))
            cnn_out = torch.reshape(cnn_out, [batch_size, -1])
            new_state_vec = self.state_net(cnn_out, state_vec)
            param_out = self.rnn_parameter_projection(new_state_vec)
            vx, vy, vrx, vry = torch.unbind(param_out, dim=-1)
            state_vec = new_state_vec
            pred_img = fft_translation(img, vx, vy)
            new_state = (state_vec, img)
        else:
            batch_size = img.shape[0]
            state_vec, prev_img = state
            centroid = compute_2d_centroid(img)
            cx = centroid[:, 0]
            cy = centroid[:, 1]
            vy, vx, _ = register_translation(img, prev_img)
            cnn_out = self.cnn(torch.stack([img, prev_img], -3))
            cnn_param_out = self.cnn_parameter_projection(cnn_out)
            ccx, ccy, cvx, cvy = torch.unbind(cnn_param_out, dim=-1)
            cx += ccx
            cy += ccy
            vx += cvx
            vy += cvy
            net_in = torch.cat([cx.unsqueeze(-1),
                    cy.unsqueeze(-1),
                    vx.unsqueeze(-1),
                    vy.unsqueeze(-1)], dim=-1)
            new_state_vec = self.state_net(net_in, state_vec)
            param_out = self.rnn_parameter_projection(new_state_vec)
            vvx, vvy, vrx, vry = torch.unbind(param_out, dim=-1)
            vx += vvx
            vy += vvy
            state_vec = new_state_vec
            pred_img = fft_translation(img, vx, vy)
            new_state = (state_vec, img)
        return pred_img, new_state


if __name__ == '__main__':
    it = MovingMNISTAdvancedIterator(initial_velocity_range=(0.0, 0.0),
                                     rotation_angle_range=(5, 5),
                                     global_rotation_angle_range=(5, 5))
    time = 7
    seq_np, motion_vectors = it.sample(5, time)
    seq_np = seq_np/255.
    seq = torch.from_numpy(seq_np[:, :, 0, :, :].astype(np.float32)).cuda()
    seq = seq[:, 0, :, :].unsqueeze(1)
    cell = RegistrationCell(learn_param_net=False, rotation=True).cuda()
    # cell = VelocityEstimationCell(cnn_depth_lst=[50, 50, 50]).cuda()
    # cell = GatedRecurrentUnitWrapper(state_size=100).cuda()
    out_lst = []
    zero_state = (torch.zeros([1, 100]).cuda(), seq[0, :, :, :])
    img = seq[1, :, :, :]
    for t in range(time - 1):
        img, new_state = cell.forward(img, zero_state)
        # plt.imshow(img[0, :, :].numpy()); plt.show()
        out_lst.append(img)

    video = torch.stack(out_lst)
    write = np.concatenate([video.cpu().numpy(), seq_np[1:, 0]], -2)
    # write = np.abs(write)/np.max(np.abs(write))
    # video_writer = VideoWriter(height=64, width=128)
    # video_writer.write_video(write[:, 0, :, :], filename='net_out.mp4')
    write_to_figure(write[:, 0, :, :])
    plt.show()

