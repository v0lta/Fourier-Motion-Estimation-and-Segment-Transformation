import pickle
import time as pytime
import torch
import numpy as np
import matplotlib.pyplot as plt
from util.write_movie import VideoWriter, write_to_figure
from cells.registration_cell import RegistrationCell, VelocityEstimationCell, GatedRecurrentUnitWrapper
from moving_mnist_pp.movingmnist_iterator import MovingMNISTAdvancedIterator


rotation = 5
it = MovingMNISTAdvancedIterator(rotation_angle_range=(-rotation, rotation),
                                 global_rotation_angle_range=(rotation, rotation))
batch_size = 200
time = 10
context_time = 4
pred_time = 6
state_size = 100
cell = RegistrationCell(state_size=state_size, rotation=True).cuda()
logdir_reg = None
logdir_gru = None

pickle.load()

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
