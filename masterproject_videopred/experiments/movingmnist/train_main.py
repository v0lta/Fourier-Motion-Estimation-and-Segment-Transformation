import sys
sys.path.insert(0, "../../")
import tensorflow as tf
import numpy as np
import time
from networks.customcell import *
from networks.encoder_decoder import EncoderDecoder
from networks.encoder import *
from networks.decoder import *
from networks.trainers import Trainer
from networks.ops import preprocess_data
from nowcasting.movingmnist_iterator import MovingMNISTAdvancedIterator
import argparse
from datetime import datetime
import os

parser = argparse.ArgumentParser(description="run the model")
parser.add_argument("-bs", "--batch_size", help="batch size",
                    type=int, default=2)
parser.add_argument("-e", "--epochs", help="epoch number of the training",
                    type=int, default=100)
parser.add_argument("-lr", "--learning_rate", help="learning rate",
                    type=float, default=0.001)
parser.add_argument("--seqlen", help="the total length of sequence",
                    type=int, default=16)
parser.add_argument("--pred_num", help="the length of sequence to predict",
                    type=int, default=8)
parser.add_argument("--digit_num", help="the number of digits in sequence",
                    type=int, default=2)
parser.add_argument("--logdir", help="the directory for the summary",
                    default=".")
parser.add_argument("--cell", help="the type of the cell",
                    default="TrajGRU")
parser.add_argument("--network", help="the architecture of the network",
                    choices=['simple', 'tcn'], default='tcn')
parser.add_argument("--encoder_rate", help="the compress rate of the encoder",
                    type=int, default=2)
parser.add_argument("--decoder_rate", help="the compress rate of the decoder",
                    type=int, default=2)
parser.add_argument("--reconstructor", help="reconstructor True",
                    action='store_true')
parser.add_argument("--predictor", help="predictor True",
                    action='store_true')
parser.add_argument("-l", "--layers", help="the number of layers",
                    type=int, default=3)

args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate
seqlen = args.seqlen
pred_num = args.pred_num
digit_num = args.digit_num
encoder_rate = args.encoder_rate
decoder_rate = args.decoder_rate
layers = args.layers

have_reconstructor = args.reconstructor
have_predictor = args.predictor


train_total = 20
val_total = 20
test_total = 20
video_shape = [seqlen, 64, 64, 1]
ROTATION = (-30, 30)

mnt_generator = MovingMNISTAdvancedIterator(digit_num=digit_num,
                                            rotation_angle_range=ROTATION)
val_vid, _ = mnt_generator.sample(batch_size=val_total, seqlen=seqlen)
val_vid = np.clip(val_vid, 0, 255)
val_vid = preprocess_data(val_vid)


celltype = args.cell
if layers == 3:
    enc_dims = [64, 32, 16]
    enc_depth = [32, 64, 128]
    dec_dims = [16, 32, 64]
    dec_depth = [128, 64, 32]
    cells_flags = [1, 1, 0]
elif layers == 2:
    enc_dims = [64, 32]
    enc_depth = [24, 48]
    dec_dims = [32, 64]
    dec_depth = [48, 24]
    #dec_outp = [None, video_shape[-1]]
    cells_flags = [1, 0]
else:
    enc_dims = [64]
    enc_depth = [4]
    dec_dims = [64]
    dec_depth = [4]
    #dec_outp = [video_shape[-1]]    

LINKS_NUM = 1
OBJECTS_NUM = 2
#ACTIVATION = tf.nn.leaky_relu
ACTIVATION = tf.nn.tanh
KERNEL_SIZE = [3, 3] 
#KERNEL_SIZE = [5, 5]

print("network_structure ", args.network, "cell_type", args.cell,
      "links_num ", LINKS_NUM, "kernel_size", KERNEL_SIZE,
      "layers ", layers, "batch_size ", batch_size,
      "train_total ", train_total,
      "epochs ", epochs, "seqlen ", seqlen, "pred_num ", pred_num,
      "reconstructor ", have_reconstructor,
      "predictor ", have_predictor,
      "rotation_angle_range ", ROTATION,
      "learning_rate ", learning_rate, "encoder_structure ", enc_depth,
      "activation ", ACTIVATION, flush=True)


def get_cell(celltype, depth, inp_dims=[64, 64], outp_depth=None,
             activation=ACTIVATION):
    if celltype == "TrajGRU":
        return TrajGRU(links_num=LINKS_NUM, kernel_size=[3, 3], depth=depth,
                       input_dims=inp_dims, output_depth=outp_depth,
                       activation=activation)
    elif celltype == "ConvGRU":
        return ConvGRU(kernel_size=KERNEL_SIZE, depth=depth,
                       input_dims=inp_dims, output_depth=outp_depth,
                       activation=activation)
    elif celltype == "FourierGRU":
        return FourierGRU(objects_num=OBJECTS_NUM, kernel_size=[3, 3], depth=depth,
                          input_dims=inp_dims, output_depth=outp_depth,
                          activation=activation)
    elif celltype == "FlowGRU":
        return FlowGRU(kernel_size=[3, 3], depth=depth,
                       input_dims=inp_dims, output_depth=outp_depth,
                       activation=activation)
    else:
        print("unkown cell type %s" % celltype)


if args.network == 'tcn':
    dec_outp = [i * encoder_rate for i in enc_depth[-2::-1]] + [video_shape[-1]]
else:
    dec_outp = [None] * (layers - 1) + [video_shape[-1]]
print("output_depth of decoder", dec_outp, flush=True)

enc_cells = [get_cell(celltype, i, [j, j]) for i, j in
             zip(enc_depth, enc_dims)]

reconstructor_cells = [get_cell(celltype, i, [j, j], k) for i, j, k in
                       zip(dec_depth, dec_dims, dec_outp)]

predictor_cells = [get_cell(celltype, i, [j, j], k) for i, j, k in
                   zip(dec_depth, dec_dims, dec_outp)]

if args.network == 'tcn':
    encoder = TemporalCompressEncoder(enc_cells, cells_flags,
                                      compress_rate=encoder_rate)
    if have_reconstructor is not True:
        reconstructor = None
    else:
        reconstructor = TemporalCompressDecoder(reconstructor_cells,
                                                cells_flags,
                                                output_length=seqlen - pred_num,
                                                compress_rate=decoder_rate)
    if have_predictor is not True:
        predictor = None
    else:
        predictor = TemporalCompressDecoder(predictor_cells, cells_flags,
                                            output_length=pred_num,
                                            compress_rate=decoder_rate)
else:
    encoder = SimpleEncoder(enc_cells)
    if have_reconstructor is not True:
        reconstructor = None
    else:
        reconstructor = SimpleDecoder(reconstructor_cells,
                                      output_length=pred_num)
    if have_predictor is not True:
        predictor = None
    else:
        predictor = SimpleDecoder(predictor_cells, output_length=pred_num)

network = EncoderDecoder(encoder, reconstructor, predictor)

dirpath = args.logdir
celltype = enc_cells[0].__class__.__name__
timestr = datetime.now().strftime('%m%d-%H%M%S')
dirname = celltype + timestr
dirpath = os.path.join(dirpath, dirname)
dirpath = os.path.normpath(dirpath)
os.makedirs(dirpath, exist_ok=True)

loss_list = []

print("initializing trainer", flush=True)
start = time.time()
trainer = Trainer(network, batch_size, video_shape, learning_rate, pred_num)
#trainer = Trainer(network, batch_size, video_shape, learning_rate, pred_num,
#                  gdl_weight=0.5)
end = time.time()
print("trainer initialized in ", end-start, flush=True)

summpath = os.path.join(dirpath, "logdir")
modelpath = os.path.join(dirpath, "model")

trainer.create_writer(summpath)

early_stopping = False
last_improvment = 0
best_loss = 100
stop_threshold = 10
save_output = True
total_time = 0
with tf.Session(graph=trainer._graph) as sess:
    saver = tf.train.Saver()
    trainer.initialize()
    param_num = trainer.get_param_count()
    print("the number of parameters: ", param_num, flush=True)
    for i in range(epochs):
        train_vid, _ = mnt_generator.sample(batch_size=train_total,
                                            seqlen=seqlen)
        train_vid = np.clip(train_vid, 0, 255)
        train_vid = preprocess_data(train_vid)
        start = time.time()
        trainer.update(train_vid, sess, i, collect_metadata=True)
        end = time.time()
        total_time += end - start
        one_epoch_time = end - start
        val_loss = trainer.get_model_output(val_vid, sess, i, mode='val')
        print("epoch %4d done, it took %6.2f, loss is %f." %(i, one_epoch_time,
                                                             val_loss),
              flush=True)
        if early_stopping is True:
            if val_loss < best_loss:
                best_loss = val_loss
                last_improvment = i
            if i - last_improvment > stop_threshold:
                print("No improvement found in a while, stopping optimization.",
                      flush=True)
                break
            #if (i + 1) % 5 == 0:
            #    saver.save(sess, modelpath, global_step=i + 1)
    print("training done.", flush=True)
    test_vid = np.load(str(seqlen) + '-testdata.npy')
    loss, reconstruct, past_gt, future_pred, future_gt = \
                                   trainer.get_model_output(test_vid,
                                                            sess,
                                                            mode='test')
    print("loss of test dataset", loss, flush=True)
    if save_output is True:
        np.savez(dirpath +  "/test_outputs", loss=loss,
                 reconstruct=reconstruct, past_gt=past_gt,
                 future_pred=future_pred, future_gt=future_gt)
total_time_in_hour = total_time / 3600.0
print("total time: %6.2f h" % total_time_in_hour, flush=True)
