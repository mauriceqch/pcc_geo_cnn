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
import multiprocessing
from tqdm import tqdm

np.random.seed(42)
tf.set_random_seed(42)

# Use CPU
# For unknown reasons, this is 3 times faster than GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

################################################################################
### Script
################################################################################
TYPE = np.uint16
DTYPE = np.dtype(TYPE)
SHAPE_LEN = 3
def load_compressed_file(file):
    with open(file, "rb") as f:
        x_shape = np.frombuffer(f.read(DTYPE.itemsize * SHAPE_LEN), dtype=TYPE)
        y_shape = np.frombuffer(f.read(DTYPE.itemsize * SHAPE_LEN), dtype=TYPE)
        string = f.read()

        return x_shape, y_shape, string

def load_compressed_files(files, batch_size=32):
    files_len = len(files)

    with multiprocessing.Pool() as p:
        logger.info('Loading data into memory (parallel reading)')
        data = np.array(list(tqdm(p.imap(load_compressed_file, files, batch_size), total=files_len)))

    return data

def input_fn(features, batch_size):
    with tf.device('/cpu:0'):
        zero = tf.constant(0)
        dataset = tf.data.Dataset.from_generator(lambda: features, (tf.string))
        dataset = dataset.map(lambda t: (t, zero))
        dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='decompress.py',
        description='Decompress a file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input_dir',
        help='Input directory.')
    parser.add_argument(
        'input_pattern',
        help='Mesh detection pattern.')
    parser.add_argument(
        'output_dir',
        help='Output directory.')
    parser.add_argument(
        'checkpoint_dir',
        help='Directory where to save/load model checkpoints.')
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size.')
    parser.add_argument(
        '--read_batch_size', type=int, default=1,
        help='Batch size for parallel reading.')
    parser.add_argument(
        '--num_filters', type=int, default=32,
        help='Number of filters per layer.')
    parser.add_argument(
        '--preprocess_threads', type=int, default=16,
        help='Number of CPU threads to use for parallel decoding.')
    parser.add_argument(
        '--output_extension', default='.ply',
        help='Output extension.')

    args = parser.parse_args()

    assert args.batch_size > 0, 'batch_size must be positive'

    DATA_FORMAT = 'channels_first'

    args.input_dir = os.path.normpath(args.input_dir)
    len_input_dir = len(args.input_dir)
    assert os.path.exists(args.input_dir), "Input directory not found"

    input_glob = os.path.join(args.input_dir, args.input_pattern)
    files = pc_io.get_files(input_glob)
    assert len(files) > 0, "No input files found"
    filenames = [x[len_input_dir+1:] for x in files]
    output_files = [os.path.join(args.output_dir, x + '.ply') for x in filenames]

    compressed_data = load_compressed_files(files, args.read_batch_size)
    x_shape = compressed_data[0][0]
    y_shape = compressed_data[0][1]
    assert np.all([np.all(x[0] == x_shape) for x in compressed_data]), 'All x_shape must be equal'
    assert np.all([np.all(x[1] == y_shape) for x in compressed_data]), 'All y_shape must be equal'
    compressed_strings = (x[2] for x in compressed_data)

    estimator = tf.estimator.Estimator(
        model_fn=compression_model.model_fn,
        model_dir=args.checkpoint_dir,
        params={
            'num_filters': args.num_filters,
            'checkpoint_dir': args.checkpoint_dir,
            'data_format': DATA_FORMAT,
            'decompress': True,
            'x_shape': x_shape,
            'y_shape': y_shape
        })
#    hook = tf.train.ProfilerHook(save_steps=1, output_dir='./decompress_profiler')
    result = estimator.predict(
            input_fn=lambda: input_fn(compressed_strings, args.batch_size),
            predict_keys=['x_hat_quant'])
#            hooks=[hook])

    len_files = len(files)
    i = 0
    for ret, ori_file, output_file in zip(result, files, output_files):
        logger.info(f'{i}/{len_files} - Writing {ori_file} to {output_file}')
        output_dir, _ = os.path.split(output_file)
        os.makedirs(output_dir, exist_ok=True)

        pa = np.argwhere(ret['x_hat_quant']).astype('float32')
        pc_io.write_df(output_file, pc_io.pa_to_df(pa))
        i += 1


