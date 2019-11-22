import pickle
import time as pytime
import torch
import numpy as np
import matplotlib.pyplot as plt
from util.write_movie import VideoWriter, write_to_figure
from cells.registration_cell import RegistrationCell, VelocityEstimationCell, GatedRecurrentUnitWrapper
from moving_mnist_pp.movingmnist_iterator import MovingMNISTAdvancedIterator
from torch.utils.tensorboard import SummaryWriter

rotation = 0
it = MovingMNISTAdvancedIterator()
batch_size = 400
time = 10
context_time = 4
pred_time = 6
state_size = 1024
# cell = RegistrationCell(state_size=state_size).cuda()
# cell = VelocityEstimationCell(cnn_depth_lst=[10, 10, 10, 10], state_size=state_size).cuda()
cell = GatedRecurrentUnitWrapper(state_size=state_size)
iterations = 10000
lr = 0.0005
opt = torch.optim.Adam(cell.parameters(), lr=lr)
grad_clip_norm = 400
criterion = torch.nn.MSELoss()
writer = torch.utils.tensorboard.writer.SummaryWriter(
    comment='_rot_' + str(rotation) + '_bs_' + str(batch_size)
            + 'clip_' + str(grad_clip_norm) + '_lr_' + str(lr)
            + '_state_' + str(state_size) + '_' + type(cell).__name__)
loss_lst = []
grad_lst = []

#with torch.autograd.detect_anomaly():
for i in range(iterations):
        # training loop
        time_start = pytime.time()
        opt.zero_grad()
        seq_np, motion_vectors = it.sample(batch_size, time)
        seq = torch.from_numpy(seq_np[:, :, 0, :, :].astype(np.float32)).cuda()
        seq = seq/255.0
        # seq = seq[:, 0, :, :].unsqueeze(1)
        context = seq[:context_time, :, :, :]
        prediction = seq[context_time:, :, :, :]
        out_lst = []
        state = (torch.zeros([batch_size, state_size]).cuda(), context[0, :, :, :])
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
        prediction = torch.clamp(prediction, 0.0, 1.0)
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
        time_end = pytime.time() - time_start
        print('it', i, 'of', iterations, 'mse', loss.detach().cpu().numpy(), 'grad-norm', total_norm, 'it-time [s]', time_end)
        loss_lst.append(loss.detach().cpu().numpy())
        grad_lst.append(total_norm)
        writer.add_scalar('loss', loss, global_step=i)
        writer.add_scalar('grad_norm', total_norm, global_step=i)
        cat_img = torch.cat([pred_vid[:, 0, :, :], prediction[:, 0, :, :]], -1)
        writer.add_image('pred_gt', cat_img[0].unsqueeze(0)/torch.max(cat_img[0]),
                         global_step=i)
        cat_img_cats = cat_img[0]
        for j in range(1, pred_time):
            cat_img_cats = torch.cat([cat_img_cats, cat_img[j]], -2)
        writer.add_image('pred_vid', cat_img_cats.unsqueeze(0)/torch.max(cat_img_cats), global_step=i)


plt.plot(loss_lst)
plt.show()
plt.plot(grad_lst)
plt.show()

net_in_gt = np.concatenate([context.detach().cpu().numpy(), prediction.detach().cpu().numpy()], 0)
net_out = np.concatenate([context.detach().cpu().numpy(), pred_vid.detach().cpu().numpy()], 0)
write = np.concatenate([net_out, net_in_gt], -2)
write = np.abs(write)/np.max(np.abs(write))
for vno in range(batch_size):
    video_writer = VideoWriter(height=128, width=64)
    video_writer.write_video(write[:, vno, :, :], filename='./' + writer.log_dir + '/' + str(vno) + '.mp4')
    plt.close()
    if vno > 50:
        break

# pickle the cell
pickle.dump(cell, open('./' + writer.log_dir + '/' + 'cell.pkl', 'wb'))
print('done')
