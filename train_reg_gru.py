import torch
import numpy as np
import matplotlib.pyplot as plt
from util.write_movie import VideoWriter
from cells.registration_cell import RegistrationCell
from moving_mnist_pp.movingmnist_iterator import MovingMNISTAdvancedIterator


it = MovingMNISTAdvancedIterator()
time = 6
context_time = 2
pred_time = 4
state_size = 20
cell = RegistrationCell(state_size=state_size)
iterations = 4000
opt = torch.optim.Adam(cell.parameters(), lr=0.0001)
grad_clip_norm = 1000
criterion = torch.nn.MSELoss()

loss_lst = []
grad_lst = []

with torch.autograd.detect_anomaly():
    for i in range(iterations):
            # training loop
            opt.zero_grad()

            seq_np, motion_vectors = it.sample(5, time)
            seq = torch.from_numpy(seq_np[:, :, 0, :, :].astype(np.float32))
            # seq = seq[:, 0, :, :].unsqueeze(1)
            context = seq[:context_time, :, :, :]
            prediction = seq[context_time:, :, :, :]
            out_lst = []
            state = (torch.zeros([5, state_size]), context[0, :, :, :])
            img = context[0, :, :, :]
            for t in range(1, context_time):
                # print(t)
                _, state = cell.forward(context[t, :, :, :], state)

            prediction_video_lst = []
            pimg = context[-1, :, :, :]
            for pt in range(0, pred_time):
                # print(pt)
                pimg, state = cell.forward(pimg, state)
                prediction_video_lst.append(pimg)

            pred_vid = torch.stack(prediction_video_lst, dim=0)
            loss = criterion(pred_vid, prediction)

            # compute gradients
            loss.backward()

            total_norm = 0
            for p in cell.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            torch.nn.utils.clip_grad_norm_(cell.parameters(), grad_clip_norm)

            # apply gradients
            opt.step()
            print(i, loss.detach().numpy(), total_norm)
            loss_lst.append(loss.detach().numpy())
            grad_lst.append(total_norm)

plt.plot(loss_lst)
plt.show()
plt.plot(grad_lst)
plt.show()

video = torch.stack(out_lst)
write = np.concatenate([video.numpy(), seq_np[1:, 0]], -1)
write = np.abs(write)/np.max(np.abs(write))
video_writer = VideoWriter(height=64, width=128)
video_writer.write_video(write[:, 0, :, :], filename='net_out.mp4')
plt.show()
