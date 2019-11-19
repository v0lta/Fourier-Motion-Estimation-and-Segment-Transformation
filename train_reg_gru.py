import torch
import numpy as np
import matplotlib.pyplot as plt
from util.write_movie import VideoWriter
from cells.registration_cell import RegistrationCell
from moving_mnist_pp.movingmnist_iterator import MovingMNISTAdvancedIterator
from torch.utils.tensorboard import SummaryWriter


it = MovingMNISTAdvancedIterator()
batch_size = 50
time = 10
context_time = 4
pred_time = 6
state_size = 200
cell = RegistrationCell(state_size=state_size)
iterations = 10000
opt = torch.optim.Adam(cell.parameters(), lr=0.000001)
grad_clip_norm = 8000
criterion = torch.nn.MSELoss()

loss_lst = []
grad_lst = []

with torch.autograd.detect_anomaly():
    for i in range(iterations):
            # training loop
            opt.zero_grad()

            seq_np, motion_vectors = it.sample(batch_size, time)
            seq = torch.from_numpy(seq_np[:, :, 0, :, :].astype(np.float32))
            # seq = seq[:, 0, :, :].unsqueeze(1)
            context = seq[:context_time, :, :, :]
            prediction = seq[context_time:, :, :, :]
            out_lst = []
            state = (torch.zeros([batch_size, state_size]), context[0, :, :, :])
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

net_in_gt = np.concatenate([context.detach().numpy(), prediction.detach().numpy()], 0)
net_out = np.concatenate([context.detach().numpy(), pred_vid.detach().numpy()], 0)
write = np.concatenate([net_out, net_in_gt], -1)
write = np.abs(write)/np.max(np.abs(write))
for vno in range(batch_size):
    video_writer = VideoWriter(height=64, width=128)
    video_writer.write_video(write[:, vno, :, :], filename='./test_vids/net_out' + str(vno) + '.mp4')
    plt.close()
