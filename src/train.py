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
import numpy as np
import tensorflow as tf
import argparse
import compression_model
import pc_io
from tensorflow.python import debug as tf_debug

np.random.seed(42)
tf.set_random_seed(42)

################################################################################
### Training
################################################################################
def train():
    """Trains the model."""

    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(args.resolution)
    files = pc_io.get_files(args.train_glob)
    points = pc_io.load_points(files, p_min, p_max)

    files_cat = np.array([os.path.split(os.path.split(x)[0])[1] for x in files])
    for cat in files_cat:
        assert (cat == 'train') or (cat == 'test')
    points_train = points[files_cat == 'train']
    points_test = points[files_cat == 'test']

    assert(len(points_train) + len(points_test) == len(points))

    config = tf.estimator.RunConfig(
        keep_checkpoint_every_n_hours=1,
        save_checkpoints_secs=600,
        keep_checkpoint_max=100,
        tf_random_seed=42)
    estimator = tf.estimator.Estimator(
        model_fn=compression_model.model_fn,
        model_dir=args.checkpoint_dir,
        config=config,
        params={
            'num_filters': args.num_filters,
            'alpha': args.alpha,
            'gamma': args.gamma,
            'lmbda': args.lmbda,
            'checkpoint_dir': args.checkpoint_dir,
            'data_format': DATA_FORMAT
        })

    hooks = None
    if args.debug_address is not None:
        hooks = [tf_debug.TensorBoardDebugHook(args.debug_address)]

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: compression_model.input_fn(points_train, args.batch_size, dense_tensor_shape, args.preprocess_threads),
        max_steps=args.max_steps,
        hooks=hooks)
    val_spec = tf.estimator.EvalSpec(
        input_fn=lambda: compression_model.input_fn(points_test, args.batch_size, dense_tensor_shape, args.preprocess_threads, repeat=False),
        steps=None,
        hooks=hooks)

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
        '--preprocess_threads', type=int, default=16,
        help='Number of CPU threads to use for parallel decoding.')
    parser.add_argument(
        '--debug_address', default=None,
        help='TensorBoard debug address.')


    args = parser.parse_args()

    os.makedirs(os.path.split(args.checkpoint_dir)[0], exist_ok=True)
    assert args.resolution > 0, 'resolution must be positive'
    assert args.batch_size > 0, 'batch_size must be positive'

    DATA_FORMAT = 'channels_first'

    train()
