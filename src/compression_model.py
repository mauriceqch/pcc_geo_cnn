import tensorflow as tf
import tensorflow_compression as tfc
from focal_loss import focal_loss
import os
import numpy as np
from collections import namedtuple

def pc_to_tf(points, dense_tensor_shape):
    x = points
    x = tf.pad(x, [[0, 0], [1, 0]])
    st = tf.sparse.SparseTensor(x, tf.ones_like(x[:,0]), dense_tensor_shape)
    return st

def process_x(x, dense_tensor_shape):
    x = tf.sparse.to_dense(x, default_value=0, validate_indices=False)
    x.set_shape(dense_tensor_shape)
    x = tf.cast(x, tf.float32)
    return x

def quantize_tensor(x):
    x = tf.clip_by_value(x, 0, 1)
    x = tf.round(x)
    x = tf.cast(x, tf.uint8)
    return x

def input_fn(features, batch_size, dense_tensor_shape, preprocess_threads, repeat=True):
    # Create input data pipeline.
    with tf.device('/cpu:0'):
        zero = tf.constant(0)
        dataset = tf.data.Dataset.from_generator(lambda: iter(features), tf.int64, tf.TensorShape([None, 3]))
        dataset = dataset.map(lambda x: pc_to_tf(x, dense_tensor_shape))
        dataset = dataset.map(lambda x: process_x(x, dense_tensor_shape), num_parallel_calls=preprocess_threads)
        dataset = dataset.map(lambda t: (t, zero))
        if repeat:
            dataset = dataset.shuffle(buffer_size=len(features))
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size * 2)

    return dataset.make_one_shot_iterator().get_next()

def analysis_transform(tensor, num_filters, data_format):
    with tf.variable_scope("analysis"):
        with tf.variable_scope("layer_0"):
            layer = tf.layers.Conv3D(
                    num_filters, (9, 9, 9), strides=(2, 2, 2), padding="same",
                    use_bias=True, activation=tf.nn.relu, data_format=data_format)
            tensor = layer(tensor)


        with tf.variable_scope("layer_1"):
            layer = tf.layers.Conv3D(
                    num_filters, (5, 5, 5), strides=(2, 2, 2), padding="same",
                    use_bias=True, activation=tf.nn.relu, data_format=data_format)
            tensor = layer(tensor)
            
        with tf.variable_scope("layer_2"):
            layer = tf.layers.Conv3D(
                    num_filters, (5, 5, 5), strides=(2, 2, 2), padding="same",
                    use_bias=False, activation=None, data_format=data_format)
            tensor = layer(tensor)

    return tensor

def synthesis_transform(tensor, num_filters, data_format):
    with tf.variable_scope("synthesis"):
        with tf.variable_scope("layer_0"):
            layer = tf.layers.Conv3DTranspose(
                    num_filters, (5, 5, 5), strides=(2, 2, 2), padding="same",
                    use_bias=True, activation=tf.nn.relu, data_format=data_format)
            tensor = layer(tensor)

        with tf.variable_scope("layer_1"):
            layer = tf.layers.Conv3DTranspose(
                    num_filters, (5, 5, 5), strides=(2, 2, 2), padding="same",
                    use_bias=True, activation=tf.nn.relu, data_format=data_format)
            tensor = layer(tensor)

        with tf.variable_scope("layer_2"):
            layer = tf.layers.Conv3DTranspose(
                    1, (9, 9, 9), strides=(2, 2, 2), padding="same",
                    use_bias=True, activation=tf.nn.relu, data_format=data_format)
            tensor = layer(tensor)

    return tensor

def model_fn(features, labels, mode, params):
    params = namedtuple('Struct', params.keys())(*params.values())
    # Unused
    del labels
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    # Get training patch from dataset.
    x = features
    num_voxels = tf.cast(tf.size(x), tf.float32)
    num_occupied_voxels = tf.reduce_sum(x)

    # Build autoencoder.
    y = analysis_transform(x, params.num_filters, params.data_format)
    entropy_bottleneck = tfc.EntropyBottleneck(data_format=params.data_format)
    y_tilde, likelihoods = entropy_bottleneck(y, training=training)
    x_tilde = synthesis_transform(y_tilde, params.num_filters, params.data_format)

    # Quantize
    x_quant = quantize_tensor(x)
    x_tilde_quant = quantize_tensor(x_tilde)

    # Total number of bits divided by number of pixels.
    log_likelihoods = tf.log(likelihoods)
    train_bpv = tf.reduce_sum(log_likelihoods) / (-np.log(2) * num_voxels)
    train_mbpov = tf.reduce_sum(log_likelihoods) / (-np.log(2) * num_occupied_voxels)

    if mode == tf.estimator.ModeKeys.PREDICT:
        string = entropy_bottleneck.compress(y)
        string = tf.squeeze(string, axis=0)
        x_shape = tf.shape(x)
        y_shape = tf.shape(y)
        predictions = {
            'x_tilde': x_tilde,
            'y_tilde': y_tilde,
            'x_tilde_quant': x_tilde_quant,
            'string': string,
            'x_shape': x_shape,
            'y_shape': y_shape
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    train_fl = focal_loss(x, x_tilde, gamma=params.gamma, alpha=params.alpha)
    # The rate-distortion cost.
    train_loss = params.lmbda * train_fl + train_mbpov

    # Metrics
    train_mae = tf.reduce_mean(tf.abs(x - x_tilde))
    train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
    tp = tf.count_nonzero(x_tilde_quant * x_quant, dtype=tf.float32) / num_voxels
    tn = tf.count_nonzero((x_tilde_quant - 1) * (x_quant - 1), dtype=tf.float32) / num_voxels
    fp = tf.count_nonzero(x_tilde_quant * (x_quant - 1), dtype=tf.float32) / num_voxels
    fn = tf.count_nonzero((x_tilde_quant - 1) * x_quant, dtype=tf.float32) / num_voxels
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)
    f1_score = (2 * precision * recall) / (precision + recall)

    tf.summary.scalar("loss", train_loss)
    tf.summary.scalar("bpv", train_bpv)
    tf.summary.scalar("mbpov", train_mbpov)
    tf.summary.scalar("mse", train_mse)
    tf.summary.scalar("focal_loss", train_fl)
    tf.summary.scalar("mae", train_mae)
    tf.summary.scalar("num_occupied_voxels", num_occupied_voxels)
    tf.summary.scalar("num_voxels", num_voxels)
    tf.summary.scalar("precision_metric", precision)
    tf.summary.scalar("recall_metric", recall)
    tf.summary.scalar("accuracy_metric", accuracy)
    tf.summary.scalar("specificity_metric", specificity)
    tf.summary.scalar("f1_score_metric", f1_score)

    tf.summary.histogram("y", y)
    tf.summary.histogram("y_tilde", y_tilde)
    tf.summary.histogram("x", x)
    tf.summary.histogram("x_tilde", x_tilde)
    tf.summary.histogram("x_tilde_quant", x_tilde_quant)
    tf.summary.histogram("likelihoods", likelihoods)
    tf.summary.histogram("log_likelihoods", log_likelihoods)

    # Creates summary for the probability mass function (PMF) estimated in the
    # bottleneck.
    entropy_bottleneck.visualize()

    if mode == tf.estimator.ModeKeys.EVAL:
        precision_metric = tf.metrics.precision(x_quant, x_tilde)
        recall_metric = tf.metrics.recall(x_quant, x_tilde)
        accuracy_metric = tf.metrics.accuracy(x_quant, x_tilde)
        metrics = {
            'precision_metric': precision_metric,
            'recall_metric': recall_metric,
            'accuracy_metric': accuracy_metric,
        }

        summary_hook = tf.train.SummarySaverHook(
            save_steps=5,
            output_dir=os.path.join(params.checkpoint_dir, 'eval'),
            summary_op=tf.summary.merge_all())
        return tf.estimator.EstimatorSpec(mode, loss=train_loss, evaluation_hooks=[summary_hook], eval_metric_ops=metrics)

    # Minimize loss and auxiliary loss, and execute update op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    main_step = main_optimizer.minimize(train_loss, global_step=tf.train.get_global_step())

    aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

    train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

    return tf.estimator.EstimatorSpec(mode, loss=train_loss, train_op=train_op)


