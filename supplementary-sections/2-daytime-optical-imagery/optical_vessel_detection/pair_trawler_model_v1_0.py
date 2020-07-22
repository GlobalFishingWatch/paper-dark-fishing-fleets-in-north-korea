from __future__ import print_function
from __future__ import division
from itertools import izip, tee
import tensorflow as tf

from .pair_trawler_model_base import ModelBase, GPUInferenceModel, L1s
from .pair_trawler_tile_data import TrainTestTileData


def model_fn(features, labels, mode, params):

    size = params['size']
    char_weight = params['char_weight']
    X = features['X'] 
    Y_ = labels 

    with tf.variable_scope('train_harness'):
        global_step = tf.train.get_global_step()

        learning_rate = tf.train.exponential_decay(
                            params['lr_max'], 
                            global_step, 
                            params['lr_decay_steps'], 
                            params['lr_decay'], 
                            staircase=True)

    def layer(l, w, d, name, nonlin=tf.nn.crelu, stride=1):

        with tf.variable_scope(name):

            return tf.layers.conv2d(l, d, w,
                        use_bias = True,
                        kernel_initializer='lecun_normal',
                        bias_initializer='zeros',
                        activation = nonlin,
                        strides=(stride, stride))


    def pool(l, n, name="pool"):
        with tf.variable_scope(name):
            return tf.concat([
                layer(l, 3, n, 'convpool', stride=2),
                tf.layers.max_pooling2d(l, 3, 2)
                      ], axis=3)

    field_of_view = params['field_of_view']

    Y = X
    with tf.variable_scope('block1'): # 315
        Y = layer(Y, 7, 24, 'conv1')
        Y = layer(Y, 7, 24, 'conv2')
        Y = pool(Y, 48) # 311
    with tf.variable_scope('block2'): # 155
        Y = layer(Y, 7, 48, 'conv1')
        Y = layer(Y, 7, 48, 'conv2')
        Y = pool(Y, 96) 
    with tf.variable_scope('block3'): # 75
        Y = layer(Y, 7, 96, 'conv1')
        Y = layer(Y, 7, 96, 'conv2')
        Y = pool(Y, 192) 
    with tf.variable_scope('block4'): # 31
        Y = layer(Y, 7, 192, 'conv1')
        Y = layer(Y, 7, 192, 'conv2')
        Y = pool(Y, 384) 
    Ycommon = Y # 9

    def make_output(n, Yx=Ycommon, count=384, keep_prob=0.9):
        Yx = layer(Yx, 9, count, 'condense1')
        Yx = tf.nn.dropout(Yx, keep_prob=keep_prob)
        Yx = layer(Yx, 1, count, 'convout1')
        Yx = tf.nn.dropout(Yx, keep_prob=keep_prob)
        Yx = layer(Yx, 1, count, 'convout2')
        Yx = tf.nn.dropout(Yx, keep_prob=keep_prob)
        Yx = layer(Yx, 1, n, 'convout3', nonlin=None)
        return Yx


    with tf.variable_scope('Yisship'):
        # Use more layers for is_ship since it's most important
        Yship_logits = make_output(1, count=1024)
        Yship = tf.nn.sigmoid(Yship_logits)

    with tf.variable_scope('Ydr'):
        Ydr = make_output(1)
    with tf.variable_scope('Ydc'):
        Ydc = make_output(1)

    with tf.variable_scope('Yangle'):
        Yangle_source = make_output(2)
        Ysin2a = Yangle_source[:, :, :, 0:1]
        Ycos2a = Yangle_source[:, :, :, 1:2]
        Yangle = 0.5 * tf.atan2(Ysin2a, Ycos2a)

    with tf.variable_scope('Ylength'):
        Ylength = make_output(1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # In `PREDICT` mode we only need to return predictions.
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions={
                "is_ship": Yship[:, :, :, 0],
                "dr" : Ydr[:, :, :, 0],
                "dc" : Ydc[:, :, :, 0],
                "angle_deg" : Yangle[:, :, :, 0],
                "length_px" : Ylength[:, :, :, 0],
                "scene_id" : features['scene_id'] 
                })

    assert Yship.shape[1:] == (1, 1, 1), Yship.shape

    upsample = params['ship_upsample']

    with tf.variable_scope('train_harness'):

        Yship_ = Y_[:, :, :, :1]
        Yrc_ = Y_[:, :, :, 1:3]
        Ydr_ = Y_[:, :, :, 1:2]
        Ydc_ = Y_[:, :, :, 2:3]
        Ylength_ = Y_[:, :, :, 3:4]
        Yangle_ = Y_[:, :, :, 4:5]
        Ysin2a_ = tf.sin(2 * Yangle_) 
        Ycos2a_ = tf.cos(2 * Yangle_)

        weights = 1.0 / ((upsample - 1) * Yship_ + 1)

        ship_loss = tf.reduce_mean(weights * tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Yship_logits, 
            labels=Yship_))

        scale = tf.maximum(Ylength_, 1.0) / 10.0

        delta_rc = tf.sqrt((Ydr - Ydr_) ** 2 + (Ydc - Ydc_) ** 2)

        rc_loss = tf.reduce_mean(Yship_ * weights * (
                            L1s(delta_rc / scale) ))

        length_loss = tf.reduce_mean(Yship_ * weights * (
                            L1s((Ylength - Ylength_) / scale)  
                           ))

        angle_loss = tf.reduce_mean(Yship_ * weights * (
                                    (Ysin2a - Ysin2a_) ** 2 +
                                    (Ycos2a - Ycos2a_) ** 2 
                                   ))

        char_loss = rc_loss + length_loss + angle_loss

        loss = (ship_loss + char_weight * char_loss)
        # Pre-made estimators use the total_loss instead of the average,
        # but that's stupid so we no longer do that.

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(
                tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True),
                5)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss=loss, 
                                              global_step=global_step)

            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, train_op=train_op)


        assert mode == tf.estimator.ModeKeys.EVAL

        # Casts are probaly not necessary, but the right fix is to use at threshold functions.
        labels =      tf.cast(Yship_ > 0.5, tf.float32)
        predictions = tf.cast(Yship  > 0.5, tf.float32)

        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
        Yrc = tf.concat([Ydr, Ydc], axis=3)
        rc_rms = tf.metrics.root_mean_squared_error(labels=Yrc_, predictions=Yrc,
                                                    weights=Yship_)
        size_rms = tf.metrics.root_mean_squared_error(labels=Ylength_, predictions=Ylength,
                                                    weights=Yship_)
        angle_rms = tf.metrics.root_mean_squared_error(
            labels = tf.concat([Ysin2a_, Ycos2a_], axis=3),
            predictions = tf.concat([Ysin2a, Ycos2a], axis=3),
                                                    weights=Yship_)
        # See this post for a pretty way to do recall / precision at thresholds
        # https://stackoverflow.com/questions/46242091/tensorflow-plot-tf-metrics-precision-at-thresholds-in-tensorboard-through-eval-m
        recall = tf.metrics.recall(labels, predictions)
        precision = tf.metrics.precision(labels, predictions)
        raw_f1 = 2 / (1 / recall[0] + 1/ precision[0])
        f1 = (raw_f1, raw_f1) # F1 is already composed of averages -- shouldn't need additional OP, so use dummy.
        ship_loss_mtrc = tf.metrics.mean(ship_loss)
        angle_loss_mtrc = tf.metrics.mean(angle_loss)
        rc_loss_mtrc = tf.metrics.mean(rc_loss)
        length_loss_mtrc = tf.metrics.mean(length_loss)

        tf.summary.scalar('is_ship_accuracy', accuracy[0])
        tf.summary.scalar('is_ship_f1', f1[0])
        tf.summary.scalar('is_ship_recall', recall[0])
        tf.summary.scalar('is_ship_precision', precision[0])
        tf.summary.scalar('rc_rms', rc_rms[0])
        tf.summary.scalar('size_rms', size_rms[0])
        tf.summary.scalar('angle_rms', angle_rms[0])
        tf.summary.scalar('ship_loss', ship_loss_mtrc[0])
        tf.summary.scalar('angle_loss', angle_loss_mtrc[0])
        tf.summary.scalar('rc_loss', rc_loss_mtrc[0])
        tf.summary.scalar('length_loss', length_loss_mtrc[0])

        tf.summary.scalar('learning_rate', learning_rate)

        eval_metrics = {
            'is_ship_accuracy' : accuracy,
            'is_ship_recall' : recall,
            'is_ship_precision' : precision,
            'is_ship_f1' : f1,
            'rc_rms' : rc_rms,
            'size_rms' : size_rms,
            'angle_rms' : angle_rms,
            'ship_loss': ship_loss_mtrc,
            'angle_loss': angle_loss_mtrc,
            'rc_loss': rc_loss_mtrc,
            'size_loss': length_loss_mtrc,
            'learning_rate' : (learning_rate, learning_rate),

        }

        return tf.estimator.EstimatorSpec(
          mode=mode,
          # Report sum of error for compatibility with pre-made estimators
          loss=loss,
          eval_metric_ops=eval_metrics)



class Model(ModelBase):

    char_weight = 1
    batch_size = 32
    lr_max_per_batch_size = 0.000125
    lr_decay_steps = 20000
    lr_decay = 0.2
    steps = 60000
    field_of_view = 339

    data_source = TrainTestTileData

    @property
    def model_fn(self):
        return model_fn


class GPUInferenceModel(GPUInferenceModel):
 
    char_weight = 1
    batch_size = 32
    lr_max_per_batch_size = 0.000125
    lr_decay_steps = 20000
    lr_decay = 0.2
    steps = 60000
    field_of_view = 339

    data_source = TrainTestTileData

    @property
    def model_fn(self):
        return model_fn


if __name__ == '__main__':
    Model.run_from_main()
