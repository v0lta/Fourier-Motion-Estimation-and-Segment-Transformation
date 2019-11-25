import pickle
import time as pytime
import torch
import numpy as np
import matplotlib.pyplot as plt
from util.write_movie import VideoWriter, write_to_figure
from cells.registration_cell import RegistrationCell, VelocityEstimationCell, GatedRecurrentUnitWrapper
from moving_mnist_pp.movingmnist_iterator import MovingMNISTAdvancedIterator


rotation = 4
it = MovingMNISTAdvancedIterator(rotation_angle_range=(-rotation, rotation),
                                 global_rotation_angle_range=(rotation, rotation))
batch_size = 200
time = 10
context_time = 4
pred_time = 6
state_size = 100
log_dir_reg = 'Nov21_21-16-29_geigelsteinclip_400_lr_0.0005_bs_800_state_50_RegistrationCell'
log_dir_gru = 'Nov22_22-53-06_infcuda_rot_0_bs_400clip_400_lr_0.0005_state_512_GatedRecurrentUnitWrapper'

reg_cell = pickle.load(log_dir_reg + 'cell.pkl')
gru_cell = pickle.load(log_dir_gru + 'cell.pkl')

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
    _, reg_state = reg_cell.forward(context[t, :, :, :], state)
    _, gru_state = gru_cell.forward(context[t, :, :, :], state)

reg_prediction_video_lst = []
gru_prediction_video_lst = []
reg_pimg = context[-1, :, :, :]
gru_pimg = context[-1, :, :, :]
for pt in range(0, pred_time):
    # print(pt)
    reg_pimg, reg_state = reg_cell.forward(reg_pimg, reg_state)
    gru_pimg, gru_state = gru_cell.forward(gru_pimg, reg_state)
    reg_prediction_video_lst.append(reg_pimg)
    gru_prediction_video_lst.append(gru_pimg)


reg_pred_vid = torch.stack(reg_prediction_video_lst, dim=0)
reg_prediction = torch.clamp(prediction, 0.0, 1.0)
loss = criterion(pred_vid, prediction)
