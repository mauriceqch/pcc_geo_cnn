################################################################################
### Init
################################################################################
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

import os
from os.path import join, basename, split, exists, normpath
from glob import glob
import numpy as np
import itertools
from operator import itemgetter
from pyntcloud import PyntCloud
from collections import defaultdict
from collections import namedtuple
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import tensorflow_compression as tfc
from focal_loss import focal_loss
import functools

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D, Input, UpSampling3D, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import argparse

np.random.seed(42)
tf.set_random_seed(42)

################################################################################
### Definitions
################################################################################
class PC:
    def __init__(self, points, p_min, p_max):
        self.points = points
        self.p_max = p_max
        self.p_min = p_min
        self.data = {}

        assert np.all(p_min < p_max), f"p_min <= p_max must be true : p_min {p_min}, p_max {p_max}"
        assert np.all(points < p_max), f"points must be inferior to p_max {p_max}"
        assert np.all(points >= p_min), f"points must be superior to p_min {p_min}"

    def __repr__(self):
        return f"<PC with {self.points.shape[0]} points (p_min: {self.p_min}, p_max: {self.p_max})>"

    def is_empty(self):
        return self.points.shape[0] == 0
    
    def p_mid(self):
        p_min = self.p_min
        p_max = self.p_max
        return p_min + ((p_max - p_min) / 2.)

def df_to_pc(df, p_min, p_max):
    points = df[['x','y','z']].values
    return PC(points, p_min, p_max)

def pa_to_df(points, attr):
    df = pd.DataFrame(data={
            'x': points[:,0],
            'y': points[:,1],
            'z': points[:,2]
            })
    
    return df

def pc_to_df(pc):
    points = pc.points
    return pa_to_df(points)

def load_pc(path, p_min, p_max):
    logger.debug(f"Loading PC {path}")
    pc = PyntCloud.from_file(path)
    ret = df_to_pc(pc.points, p_min, p_max)
    logger.debug(f"Loaded PC {path}")

    return ret

def pc_to_tf(points, dense_tensor_shape):
    x = points
    x = tf.pad(x, [[0, 0], [1, 0]])
    st = tf.sparse.SparseTensor(x, tf.ones_like(x[:,0]), dense_tensor_shape)
    return st

def write_pc(path, pc):
    df = pc_to_df(pc)
    pc2 = PyntCloud(df)
    pc2.to_file(path)

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

################################################################################
### Model
################################################################################
def analysis_transform(tensor, num_filters):
    with tf.variable_scope("analysis"):
        with tf.variable_scope("layer_0"):
            layer = tf.layers.Conv3D(
                    num_filters, (9, 9, 9), strides=(2, 2, 2), padding="same",
                    use_bias=True, activation=tf.nn.relu, data_format=DATA_FORMAT)
            tensor = layer(tensor)


        with tf.variable_scope("layer_1"):
            layer = tf.layers.Conv3D(
                    num_filters, (5, 5, 5), strides=(2, 2, 2), padding="same",
                    use_bias=True, activation=tf.nn.relu, data_format=DATA_FORMAT)
            tensor = layer(tensor)

    return tensor

def synthesis_transform(tensor, num_filters):
    with tf.variable_scope("synthesis"):
        with tf.variable_scope("layer_0"):
            layer = tf.layers.Conv3DTranspose(
                    num_filters, (5, 5, 5), strides=(2, 2, 2), padding="same",
                    use_bias=True, activation=tf.nn.relu, data_format=DATA_FORMAT)
            tensor = layer(tensor)

        with tf.variable_scope("layer_1"):
            layer = tf.layers.Conv3DTranspose(
                    1, (9, 9, 9), strides=(2, 2, 2), padding="same",
                    use_bias=True, activation=tf.nn.relu, data_format=DATA_FORMAT)
            tensor = layer(tensor)

    return tensor

def input_fn(features, batch_size, dense_tensor_shape, preprocess_threads, repeat=True):
    # Create input data pipeline.
    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_generator(lambda: iter(features), tf.int64, tf.TensorShape([None, 3]))
        dataset = dataset.map(lambda x: pc_to_tf(x, dense_tensor_shape))
        dataset = dataset.map(lambda x: process_x(x, dense_tensor_shape), num_parallel_calls=preprocess_threads)
        dataset = dataset.map(lambda t: (t, tf.constant(0)))
        if repeat:
            dataset = dataset.shuffle(buffer_size=len(features))
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size * 2)

    return dataset.make_one_shot_iterator().get_next()

def model_fn(features, labels, mode, params):
    params = namedtuple('Struct', params.keys())(*params.values())
    # Unused
    del labels
    training = mode == tf.estimator.ModeKeys.TRAIN

    # Get training patch from dataset.
    x = features
    num_voxels = tf.cast(tf.size(x), tf.float32)
    num_occupied_voxels = tf.reduce_sum(x)

    # Build autoencoder.
    y = analysis_transform(x, params.num_filters)
    entropy_bottleneck = tfc.EntropyBottleneck(data_format=DATA_FORMAT)
    y_tilde, likelihoods = entropy_bottleneck(y, training=training)
    x_tilde = synthesis_transform(y_tilde, params.num_filters)

    # Quantize
    x_quant = quantize_tensor(x)
    x_tilde_quant = quantize_tensor(x_tilde)

    # Total number of bits divided by number of pixels.
    log_likelihoods = tf.log(likelihoods)
    train_bpv = tf.reduce_sum(log_likelihoods) / (-np.log(2) * num_voxels)
    train_mbpov = tf.reduce_sum(log_likelihoods) / (-np.log(2) * num_occupied_voxels)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'x_tilde': x_tilde,
            'train_bpv': train_bpv,
            'y_tilde': y_tilde,
            'x_tilde_quant': x_tilde_quant
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    train_fl = focal_loss(x, x_tilde, gamma=params.gamma, alpha=params.alpha)
    # The rate-distortion cost.
    train_loss = params.lmbda * train_fl + train_mbpov

    # Metrics
    train_mae = tf.reduce_mean(tf.abs(x - x_tilde))
    train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
    precision_metric = tf.metrics.precision(x_quant, x_tilde)
    recall_metric = tf.metrics.recall(x_quant, x_tilde)
    accuracy_metric = tf.metrics.accuracy(x_quant, x_tilde)

    tf.summary.scalar("loss", train_loss)
    tf.summary.scalar("bpv", train_bpv)
    tf.summary.scalar("mbpov", train_mbpov)
    tf.summary.scalar("mse", train_mse)
    tf.summary.scalar("focal_loss", train_fl)
    tf.summary.scalar("mae", train_mae)
    tf.summary.scalar("num_occupied_voxels", num_occupied_voxels)
    tf.summary.scalar("num_voxels", num_voxels)
    tf.summary.scalar("precision_metric", precision_metric[1])
    tf.summary.scalar("recall_metric", recall_metric[1])
    tf.summary.scalar("accuracy_metric", accuracy_metric[1])

    tf.summary.histogram("y", y)
    tf.summary.histogram("y_tilde", y_tilde)
    tf.summary.histogram("x", x)
    tf.summary.histogram("x_tilde", x_tilde)
    tf.summary.histogram("x_tilde_quant", x_tilde_quant)
    tf.summary.histogram("likelihoods", likelihoods)
    tf.summary.histogram("log_likelihoods", log_likelihoods)

    tf.summary.tensor_summary("original", x_quant)
    tf.summary.tensor_summary("reconstruction", x_tilde_quant)

    # Creates summary for the probability mass function (PMF) estimated in the
    # bottleneck.
    entropy_bottleneck.visualize()

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
            'precision_metric': precision_metric,
            'recall_metric': recall_metric,
            'accuracy_metric': accuracy_metric,
        }

        summary_hook = tf.train.SummarySaverHook(
            save_steps=5,
            output_dir=join(params.checkpoint_dir, 'eval'),
            summary_op=tf.summary.merge_all())
        return tf.estimator.EstimatorSpec(mode, loss=train_loss, evaluation_hooks=[summary_hook], eval_metric_ops=metrics)

    # Minimize loss and auxiliary loss, and execute update op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    main_step = main_optimizer.minimize(train_loss, global_step=tf.train.get_global_step())

    aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

    train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])
    train_op = main_step

    return tf.estimator.EstimatorSpec(mode, loss=train_loss, train_op=train_op)



################################################################################
### Training
################################################################################
def train():
    """Trains the model."""

    bbox_min = 0
    bbox_max = args.resolution
    p_max = np.array([bbox_max, bbox_max, bbox_max])
    p_min = np.array([bbox_min, bbox_min, bbox_min])
    dense_tensor_shape = np.concatenate([[1], p_max]).astype('int64')
 

    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    # Load data to memory
    files = np.array(glob(args.train_glob, recursive=True))
    files_len = len(files)
    files_cat = np.array([split(split(x)[0])[1] for x in files])
    for cat in files_cat:
        assert (cat == 'train') or (cat == 'test')

    with Pool() as p:
        logger.info('Loading PCs into memory (parallel reading)')
        pcs = np.array(list(tqdm(p.imap(functools.partial(load_pc, p_min=p_min, p_max=p_max), files, 32), total=files_len)))
        pcs_tf = np.array(list(tqdm((pc.points for pc in pcs), total=files_len)))

    points = pcs_tf
    points_train = points[files_cat == 'train']
    points_test = points[files_cat == 'test']
    points_val = points_test
    # points_train, points_val = train_test_split(points_train, test_size=0.2)

    # assert(len(points_train) + len(points_val) + len(points_test) == len(points))
    assert(len(points_train) + len(points_val) == len(points))

    config = tf.estimator.RunConfig(
        keep_checkpoint_every_n_hours=1,
        save_checkpoints_secs=600,
        keep_checkpoint_max=100,
        tf_random_seed=42)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.checkpoint_dir,
        config=config,
        params={
            'num_filters': args.num_filters,
            'alpha': args.alpha,
            'gamma': args.gamma,
            'lmbda': args.lmbda,
            'checkpoint_dir': args.checkpoint_dir,
        })
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(points_train, args.batch_size, dense_tensor_shape, args.preprocess_threads),
        max_steps=args.max_steps)
    val_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(points_val, args.batch_size, dense_tensor_shape, args.preprocess_threads, repeat=False),
        steps=None)

    tf.estimator.train_and_evaluate(estimator, train_spec, val_spec)

################################################################################
### Script
################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='train.py',
        description='Train network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'train_glob',
        help='Glob pattern for identifying training data.')
    parser.add_argument(
        'checkpoint_dir',
        help='Directory where to save/load model checkpoints.')
    parser.add_argument(
        '--resolution',
        type=int, help='Dataset resolution.', default=64)
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Report bitrate and distortion when training.')
    parser.add_argument(
        '--num_filters', type=int, default=32,
        help='Number of filters per layer.')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size for training.')
    parser.add_argument(
        '--lmbda', type=float, default=0.0001,
        help='Lambda for rate-distortion tradeoff.')
    parser.add_argument(
        '--alpha', type=float, default=0.9,
        help='Focal loss alpha.')
    parser.add_argument(
        '--gamma', type=float, default=2.0,
        help='Focal loss gamma.')
    parser.add_argument(
        '--max_steps', type=int, default=1000000,
        help='Train up to this number of steps.')
    parser.add_argument(
        '--epochs', type=int, default=1000,
        help='Train up to this number of epochs.')
    parser.add_argument(
        '--preprocess_threads', type=int, default=16,
        help='Number of CPU threads to use for parallel decoding.')

    args = parser.parse_args()

    os.makedirs(split(args.checkpoint_dir)[0], exist_ok=True)
    assert args.resolution > 0, 'resolution must be positive'
    assert args.batch_size > 0, 'batch_size must be positive'

    DATA_FORMAT = 'channels_first'

    train()
