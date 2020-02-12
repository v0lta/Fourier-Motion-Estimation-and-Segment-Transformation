# Created by moritz (wolter@cs.uni-bonn.de) at 28/11/2019

import pickle
import time as pytime
import torch
import numpy as np
import matplotlib.pyplot as plt
from util.write_movie import VideoWriter, write_to_figure
from cells.registration_cell import RegistrationCell, VelocityEstimationCell, GatedRecurrentUnitWrapper
from moving_mnist_pp.movingmnist_iterator import MovingMNISTAdvancedIterator

rotation = 4
it = MovingMNISTAdvancedIterator(initial_velocity_range=(2.25, 2.25),
                                 rotation_angle_range=(rotation, rotation),
                                 global_rotation_angle_range=(rotation, rotation))
batch_size = 550
time = 10
context_time = 4
pred_time = 6
reg_state_size = 100
gru_state_size = 512
runs_dir = './runs/'
log_dir_reg = 'retest/Nov26_16-42-01_teufelskapelle_rot_4_bs_550_clip_3_lr_0.0005_' \
              'state_100_RegistrationCell_retest'
log_dir_gru = 'retest/Nov27_16-57-23_teufelskapelle_rot_4_bs_550_clip_3_lr_0.0005_' \
              'state_512_GatedRecurrentUnitWrapper_retest'

reg_cell = pickle.load(open(runs_dir + log_dir_reg + '/cell.pkl', 'rb'))
gru_cell = pickle.load(open(runs_dir + log_dir_gru + '/ir_5000_cell.pkl', 'rb'))

reg_cell_params = 0
for parameter in reg_cell.parameters():
    print('reg param', parameter.shape)
    reg_cell_params += np.prod(parameter.shape)
print('reg total', reg_cell_params)

gru_cell_params = 0
for parameter in gru_cell.parameters():
    print('gru param', parameter.shape)
    gru_cell_params += np.prod(parameter.shape)
print('gru params', gru_cell_params)


seq_np, motion_vectors = it.sample(batch_size, time)

seq = torch.from_numpy(seq_np[:, :, 0, :, :].astype(np.float32)).cuda()
seq = seq/255.0
# seq = seq[:, 0, :, :].unsqueeze(1)
context = seq[:context_time, :, :, :]
prediction = seq[context_time:, :, :, :]
out_lst = []
reg_state = (torch.zeros([batch_size, reg_state_size]).cuda(), context[0, :, :, :])
gru_state = (torch.zeros([batch_size, gru_state_size]).cuda(), context[0, :, :, :])
img = context[0, :, :, :]
for t in range(1, context_time):
    _, reg_state = reg_cell.forward(context[t, :, :, :], reg_state)
    _, gru_state = gru_cell.forward(context[t, :, :, :], gru_state)

reg_prediction_video_lst = []
gru_prediction_video_lst = []
reg_pimg = context[-1, :, :, :]
gru_pimg = context[-1, :, :, :]
for pt in range(0, pred_time):
    # print(pt)
    reg_pimg, reg_state = reg_cell.forward(reg_pimg, reg_state)
    gru_pimg, gru_state = gru_cell.forward(gru_pimg, gru_state)
    reg_prediction_video_lst.append(reg_pimg)
    gru_prediction_video_lst.append(gru_pimg)


# evaluation
def kl_divergence(gt, pred):
    """
    Computes the Kullback-Leibler divergence.
    :param gt: ground truth data array [batch_size, time, height, width].
    :param pred: prediction data array [batch_size, time, height, width]
    :return: The kl-divergence between gt and pred.
    """
    assert gt.shape == pred.shape, 'we need identical shapes for comparison'
    gt_bt = np.reshape(gt, [gt.shape[0], gt.shape[1], -1])
    pred_bt = np.reshape(gt, [pred.shape[0], pred.shape[1], -1])
    gt_dist = gt_bt/np.expand_dims(np.sum(gt_bt, axis=-1), -1) + 1e-8
    pred_dist = pred_bt/np.expand_dims(np.sum(pred_bt, axis=-1), -1) + 1e-8
    kl_gp = np.sum(gt_dist*np.log(gt_dist/pred_dist))
    kl_pg = np.sum(pred_dist*np.log(pred_dist/gt_dist))
    return kl_gp, kl_pg

gt_pred_vid = prediction.cpu().numpy()
reg_pred_vid = torch.clamp(torch.stack(reg_prediction_video_lst, dim=0), 0, 1)
gru_pred_vid = torch.clamp(torch.stack(gru_prediction_video_lst, dim=0), 0, 1)
reg_context_pred_vid = torch.cat([context, reg_pred_vid], dim=0).detach().cpu().numpy()
gru_context_pred_vid = torch.cat([context, gru_pred_vid], dim=0).detach().cpu().numpy()
reg_pred_vid = reg_pred_vid.detach().cpu().numpy()
gru_pred_vid = gru_pred_vid.detach().cpu().numpy()

print('kl_div gt reg-pred', kl_divergence(gt_pred_vid, reg_pred_vid))
print('kl_div gt gru', kl_divergence(gt_pred_vid, gru_pred_vid))

print('std gt', np.std(gt_pred_vid))
print('std reg-pred', np.std(reg_pred_vid))
print('std gru-pred', np.std(gru_pred_vid))

print('mean gt', np.mean(gt_pred_vid))
print('mean reg-pred', np.mean(reg_pred_vid))
print('mean gru-pred', np.mean(gru_pred_vid))

print('mse reg-pred', np.mean(np.square(gt_pred_vid - reg_pred_vid)))
print('mse gru-pred', np.mean(np.square(gt_pred_vid - gru_pred_vid)))

gt = seq[:, :, :, :].detach().cpu().numpy()
black_bar = np.ones([gt.shape[0], gt.shape[1], 1, 64])
write_array = np.concatenate([gt, black_bar, reg_context_pred_vid, black_bar, gru_context_pred_vid], -2)
black_bar = np.ones([gt.shape[0], gt.shape[1], 64, 1])
write_video_array = np.concatenate([gt, black_bar, reg_context_pred_vid, black_bar, gru_context_pred_vid], -1)
for batch_no in range(write_array.shape[1]):
    video_writer = VideoWriter(height=64, width=194)
    video_writer.write_video(write_video_array[:, batch_no, :, :], 'result' + str(batch_no) + '.mp4')
    write_to_figure(write_array[:, batch_no, :, :], labels=True)
    plt.show()
    if batch_no > 10:
        break