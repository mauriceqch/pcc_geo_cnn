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
import subprocess
import pandas as pd
from tqdm import tqdm
from pyntcloud import PyntCloud

np.random.seed(42)
tf.set_random_seed(42)

################################################################################
### Script
################################################################################
def pc_error(f, df):
        command = f'{args.pc_error_path} -a {f} -b {df} -d --knn {args.pc_error_knn}'
        logger.info("Executing " + command)
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        decoded_output = output.decode(args.pc_error_output_encoding).split('\n')
        data_lines = [x for x in decoded_output if args.pc_error_data_signal in x]
        parsed_data_lines = [x[len(args.pc_error_data_signal):] for x in data_lines]
        # Before last value : information about the metric
        # Last value : metric value
        data = [(','.join(x[:-1]), x[-1]) for x in [x.split(',') for x in parsed_data_lines]]

        return data

def pc_error_packed(d):
    return pc_error(d[0], d[1])

def get_n_points(f):
    return len(PyntCloud.from_file(f).points)

def get_file_size_in_bits(f):
    return os.stat(f).st_size * 8

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='eval.py',
        description='Outputs evaluation results from original, compressed and decompressed point cloud folders.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'ori_dir',
        help='Original directory.')
    parser.add_argument(
        'ori_pattern',
        help='Mesh detection pattern.')
    parser.add_argument(
        'decompressed_dir',
        help='Decompressed directory.')
    parser.add_argument(
        'pc_error_path',
        help='Path to pc_error executable.')
    parser.add_argument(
        '--decompressed_suffix',
        help='Decompressed suffix.',
        default="")
    parser.add_argument(
        '--compressed_dir',
        help='Compressed directory.')
    parser.add_argument(
        '--compressed_suffix',
        help='Compressed suffix.',
        default="")
    parser.add_argument(
        '--output_file',
        help='Output path for report.',
        default='./eval.csv')
    parser.add_argument(
        '--pc_error_knn',
        type=int,
        help='pc_error knn argument.',
        default=36)
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Parallelism batch size.',
        default=1)
    parser.add_argument(
        '--pc_error_data_signal',
        help='pc_error data signal for parsing.',
        default='   ### ')
    parser.add_argument(
        '--pc_error_output_encoding',
        help='pc_error output encoding.',
        default='utf-8')

    args = parser.parse_args()

    logger.info("Checking pc_error_path")
    assert os.path.exists(args.pc_error_path)

    args.ori_dir = os.path.normpath(args.ori_dir)
    args.decompressed_dir = os.path.normpath(args.decompressed_dir)

    assert os.path.exists(args.ori_dir), "Original directory not found"
    assert os.path.exists(args.decompressed_dir), "Decompressed directory not found"

    compressed_dir_supplied = args.compressed_dir is not None
    if compressed_dir_supplied:
        args.compressed_dir = os.path.normpath(args.compressed_dir)
        assert os.path.exists(args.compressed_dir), "Compressed directory not found"

    logger.info("Loading files list")
    ori_glob = os.path.join(args.ori_dir, args.ori_pattern)
    files = pc_io.get_files(ori_glob)
    assert len(files) > 0, "No ori files found"
    filenames = [x[len(args.ori_dir)+1:] for x in files]
    if compressed_dir_supplied:
        compressed_files = [os.path.join(args.compressed_dir, x + args.compressed_suffix) for x in filenames]
    decompressed_files = [os.path.join(args.decompressed_dir, x + args.decompressed_suffix) for x in filenames]

    logger.info("Checking filenames consistency")
    for f, df in zip(files, decompressed_files):
        assert os.path.exists(f), f + " does not exist"
        assert os.path.exists(df), df + " does not exist"

        assert f[len(args.ori_dir):] == df[len(args.decompressed_dir):len(df)-len(args.decompressed_suffix)]

    if compressed_dir_supplied:
        for f, cf in zip(files, compressed_files):
            assert os.path.exists(cf), cf + " does not exist"
            assert f[len(args.ori_dir):] == cf[len(args.compressed_dir):len(cf)-len(args.compressed_suffix)]

    headers = []
    results = []

    with multiprocessing.Pool() as p:
        logger.info("Computing quality metrics")
        data_list = np.array(list(tqdm(p.imap(pc_error_packed, zip(files, decompressed_files), args.batch_size), total=len(files))))
        logger.info("Getting number of points")
        n_points_ori = np.array(list(tqdm(p.imap(get_n_points, files, args.batch_size), total=len(files))))
        n_points_out = np.array(list(tqdm(p.imap(get_n_points, decompressed_files, args.batch_size), total=len(decompressed_files))))
        if compressed_dir_supplied:
            logger.info("Getting compressed file sizes in bits")
            csib = np.array(list(tqdm(p.imap(get_file_size_in_bits, compressed_files, args.batch_size), total=len(files))))

    for data in data_list:
        if len(headers) == 0:
            headers = [x[0] for x in data]
        results_row = [float(x[1]) for x in data]
        assert len(headers) == len(results_row), "Inconsistent output from pc_error"
        results.append(results_row)

    df = pd.DataFrame(data=results, columns=headers)

    df['ori_file'] = files
    df['decompressed_file'] = decompressed_files
    df['n_points_ori'] = n_points_ori
    df['n_points_out'] = n_points_out
    if compressed_dir_supplied:
        logger.info("Getting compressed file sizes")
        df['compressed_file'] = compressed_files
        df['compressed_size_in_bits'] = csib
        df['bpov'] = df['compressed_size_in_bits'] / df['n_points_ori']

    df.to_csv(args.output_file)
 
