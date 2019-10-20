import tensorflow as tf
import numpy as np
import ipdb


class Trainer():
    """A trainer used to train a network.
    """

    def __init__(self, model_graph, batch_size, video_shape, learning_rate,
                 pred_num, gdl_weight=None):
        """
        Args:
            model_graph: A model graph where the weights will be optimized.
            batch_size: the number of sequences per batch.
            video_shape: the shape of video
                         [time, height, width, channels]
            learning_rate: the step size for the gradient descent algorithm.
            pred_num: the number of the future frames.
        """
        self._model_graph = model_graph
        self._batch_size = batch_size
        self._video_shape = video_shape
        self._pred_num = pred_num
        self._train_writer = None
        self._val_writer = None
        self._graph = tf.Graph()

        with self._graph.as_default():
            self._global_step = tf.Variable(0, name='global_step',
                                            trainable=False)
            self._train_vid = tf.placeholder(tf.float32,
                                             shape=(video_shape[0], batch_size,
                                                    video_shape[1],
                                                    video_shape[2], video_shape[3]))
            past_num = video_shape[0] - pred_num
            self._past_gt, self._future_gt = tf.split(self._train_vid,
                                                      [past_num, pred_num], 0)
            self._reconstruct, self._future_pred = \
                self._model_graph(self._past_gt)
            self._loss = 0
            if self._reconstruct is not None:
                self._reconstruct_loss = self._ms_loss(self._past_gt,
                                                       self._reconstruct,
                                                       "reconstruct_loss")
                self._loss += self._reconstruct_loss
                self._r_loss_summ = tf.summary.scalar("reconstruct_loss",
                                                      self._reconstruct_loss)
            if self._future_pred is not None:
                self._pred_loss = self._ms_loss(self._future_gt,
                                                self._future_pred,
                                                "pred_loss")
                self._loss += self._pred_loss
                self._p_loss_summ = tf.summary.scalar("pred_loss",
                                                      self._pred_loss)
            if self._reconstruct is not None and \
                    self._future_pred is not None:
                self._loss /= 2
                self._t_loss_summ = tf.summary.scalar("total_loss",
                                                      self._loss)
            #loss_descrip = "ms"
            '''
            if gdl_weight is not None:
                #self._loss = (1 - gdl_weight) * self._ms_loss() + \
                #             gdl_weight * self._gdl
                #set weight for mse to be 1
                print("train with combination loss, gdl weight is ", 
                      gdl_weight, flush=True)
                self._loss = self._ms_loss() + gdl_weight * self._gdl()
                loss_descrip = "combination_loss"
            '''
            #self._loss_summ = tf.summary.scalar(loss_descrip, self._loss)
            '''
            learning_rate = tf.train.exponential_decay(learning_rate,
                                                       self._global_step,
                                                       10000,
                                                       0.9,
                                                       staircase=True)
            '''
            self._lr_summ = tf.summary.scalar("lr", learning_rate)
            # test for memory use
            optimizer = tf.train.AdamOptimizer(learning_rate)
            #optimizer = tf.train.RMSPropOptimizer(learning_rate)
            # ipdb.set_trace()
            gradients, v = zip(*optimizer.compute_gradients(self._loss))

            for grad_tensor in gradients:
                tf.summary.histogram('gradients' + grad_tensor.name, grad_tensor)

            with tf.variable_scope("clip_gradient"):
                gradients = [tf.clip_by_value(grad, -1, 1)
                             for grad in gradients]
                #gradients, _ = tf.clip_by_global_norm(gradients, 5)
            self._weight_update = \
                optimizer.apply_gradients(zip(gradients, v),
                                          global_step=self._global_step)
            self._init_op = tf.global_variables_initializer()
            # ipdb.set_trace()
            self._summ = tf.summary.merge_all()

    def _ms_loss(self, gt, generation, scope=None):
        """Mean squared loss function.

        Args:
            future_gt: true future frames.
            future_pred: future prediction from the the network.

        Returns:
            loss: the loss between ground truth and prediction.
        """
        with tf.variable_scope(scope or "pixel_wise_mse"):
            loss = tf.losses.mean_squared_error(gt, generation,
                                                reduction=tf.losses.Reduction.MEAN)
            return loss

    '''
    def _gdl(self, alpha=2):
        """gradient difference loss.

        Args:
            alpha: alpha norm
        ref: Deep Multi-scale video prediction beyond mean square error.
            https://arxiv.org/abs/1511.05440
        """
        with tf.variable_scope("gdl"):
            time, bs, h, w, c = self._future_gt.get_shape().as_list()
            gt = tf.reshape(self._future_gt, (-1, h, w, c))
            pred = tf.reshape(self._future_pred, (-1, h, w, c))
            gt_dy, gt_dx = tf.image.image_gradients(gt)
            pred_dy, pred_dx = tf.image.image_gradients(pred)
            loss = tf.pow(tf.abs(tf.abs(gt_dx) - tf.abs(pred_dx)), alpha) +\
                   tf.pow(tf.abs(tf.abs(gt_dy) - tf.abs(pred_dy)), alpha)
            return tf.reduce_mean(loss)
    '''

    def initialize(self):
        """Initialize all the Variables in the graph.
        """
        self._init_op.run()
        print("trainer initialized...")

    def create_writer(self, path):
        """Create tensorflow summary writer.
        """
        self._train_writer = tf.summary.FileWriter(path + '/train',
                                                   self._graph)
        # the eventfile will be larger than 2GB if graph is included
        #self._train_writer = tf.summary.FileWriter(path + '/train')
        self._val_writer = tf.summary.FileWriter(path + '/validate')
        # self._train_writer.add_graph(self._graph)
        # self._val_writer.add_graph(self._graph)

    def get_param_count(self):
        """Get the number of parameters of the network.
        """
        total_parameters = 0
        # iterating over all variables
        for variable in tf.trainable_variables():
            local_parameters = 1
            shape = variable.get_shape()  # getting shape of a variable
            for i in shape:
                local_parameters *= i.value  # mutiplying dimension values
            total_parameters += local_parameters
        return total_parameters

    def update(self, video_data, session, epoch, collect_metadata=False):
        """update the network weights.

        Args:
            video_data: [maxtime, total_batches, height, width, channels].
            session: the current tensorflow session.
            epoch: the current pass over the entire dataset.
        """
        batch_no = int(video_data.shape[1] / self._batch_size)
        #batch_list = np.split(video_data, batch_no, axis=1)
        #loss_list = []
        epoch_loss = 0
        opts = None
        run_metadata = None
        if collect_metadata is True:
            opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        # for batch_count, batch in enumerate(batch_list):
        #    feed_dict = {self._train_vid:batch}
        for i in range(batch_no):
            feed_dict = {self._train_vid: video_data[:, i*self._batch_size:
                                                     (i+1)*self._batch_size]}
            fetch = [self._loss, self._summ, self._global_step,
                     self._weight_update]
            if collect_metadata is True and i == batch_no - 1 and epoch < 1:
                print("collect metadata")
                loss, summ, global_step, _ = \
                    session.run(fetch,
                                feed_dict=feed_dict,
                                options=opts,
                                run_metadata=run_metadata)
                if self._train_writer is not None:
                    self._train_writer.add_run_metadata(run_metadata,
                                                        'epoch %d' % epoch)
                    self._train_writer.add_summary(summ, global_step)
            else:
                #feed_dict = {self._train_vid:batch}
                # fetch = [self._loss, self._summ, self._global_step,
                #         self._weight_update]
                loss, summ, global_step, _ = \
                    session.run(fetch,
                                feed_dict=feed_dict)
                if self._train_writer is not None:
                    # if global_step % 10 == 0:
                    #    self._train_writer.add_summary(summ, global_step)
                    self._train_writer.add_summary(summ, global_step)

            epoch_loss += loss / batch_no

    def get_model_output(self, video_data, session, epoch=None, mode='val'):
        batch_no = int(video_data.shape[1] / self._batch_size)
        batch_list = np.split(video_data, batch_no, axis=1)
        loss_avg = 0
        #loss_list = []
        reconstruct_l = []
        past_gt_l = []
        future_pred_l = []
        future_gt_l = []
        for batch_count, batch in enumerate(batch_list):
            feed_dict = {self._train_vid: batch}
            fetch = [self._loss, self._summ]
            if self._reconstruct is not None and \
                    self._future_pred is not None:
                fetch = fetch + [self._reconstruct, self._past_gt]
                fetch = fetch + [self._future_pred, self._future_gt]
                loss, summ, reconstruct, past_gt, future_pred, future_gt = \
                    session.run(fetch, feed_dict=feed_dict)
            elif self._reconstruct is not None:
                fetch = fetch + [self._reconstruct, self._past_gt]
                loss, summ, reconstruct, past_gt = \
                    session.run(fetch, feed_dict=feed_dict)
            elif self._future_pred is not None:
                fetch = fetch + [self._future_pred, self._future_gt]
                loss, summ, future_pred, future_gt = \
                    session.run(fetch, feed_dict=feed_dict)

            loss_avg += loss / batch_no
            # loss_list.append(loss)
            if 'val' == mode:
                if self._val_writer is not None:
                    self._val_writer.add_summary(summ, epoch*batch_no +
                                                 batch_count)
            if 'test' == mode:
                if self._future_pred is not None:
                    future_pred_l.append(future_pred)
                    future_gt_l.append(future_gt)
                if self._reconstruct is not None:
                    reconstruct_l.append(reconstruct)
                    past_gt_l.append(past_gt)
        #loss_avg = np.mean(loss_list)
        outputs = loss_avg
        if 'test' == mode:
            if self._future_pred is not None:
                future_pred = np.concatenate(future_pred_l, axis=1)
                future_gt = np.concatenate(future_gt_l, axis=1)
            else:
                future_pred = None
                future_gt = None
            if self._reconstruct is not None:
                reconstruct = np.concatenate(reconstruct_l, axis=1)
                past_gt = np.concatenate(past_gt_l, axis=1)
            else:
                reconstruct = None
                past_gt = None
            outputs = loss_avg, reconstruct, past_gt, future_pred, future_gt
        return outputs
