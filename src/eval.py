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
        'compressed_dir',
        help='Compressed directory.')
    parser.add_argument(
        'decompressed_dir',
        help='Decompressed directory.')
    parser.add_argument(
        'compressed_suffix',
        help='Compressed suffix.')
    parser.add_argument(
        'decompressed_suffix',
        help='Decompressed suffix.')
    parser.add_argument(
        'pc_error_path',
        help='Path to pc_error executable.')
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
    args.compressed_dir = os.path.normpath(args.compressed_dir)
    args.decompressed_dir = os.path.normpath(args.decompressed_dir)

    logger.info("Checking directories")
    assert os.path.exists(args.ori_dir), "Original directory not found"
    assert os.path.exists(args.compressed_dir), "Compressed directory not found"
    assert os.path.exists(args.decompressed_dir), "Decompressed directory not found"

    logger.info("Loading files list")
    ori_glob = os.path.join(args.ori_dir, args.ori_pattern)
    files = pc_io.get_files(ori_glob)
    assert len(files) > 0, "No ori files found"
    filenames = [x[len(args.ori_dir)+1:] for x in files]
    compressed_files = [os.path.join(args.compressed_dir, x + args.compressed_suffix) for x in filenames]
    decompressed_files = [os.path.join(args.decompressed_dir, x + args.decompressed_suffix) for x in filenames]

    logger.info("Checking filenames consistency")
    for f, cf, df in zip(files, compressed_files, decompressed_files):
        assert os.path.exists(f), f + " does not exist"
        assert os.path.exists(cf), cf + " does not exist"
        assert os.path.exists(df), df + " does not exist"

        assert f[len(args.ori_dir):] == cf[len(args.compressed_dir):-len(args.compressed_suffix)]
        assert f[len(args.ori_dir):] == df[len(args.decompressed_dir):-len(args.decompressed_suffix)]

    headers = []
    results = []
    additional_headers = ['ori_file', 'compressed_file', 'decompressed_file', 'compressed_size_in_bits', 'n_points_ori', 'bpov']
    logger.info("Computing quality metrics")
    for f, cf, df in zip(tqdm(files), compressed_files, decompressed_files):
        command = f'{args.pc_error_path} -a {f} -b {df} -d --knn {args.pc_error_knn}'
        logger.info("Executing " + command)
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        decoded_output = output.decode(args.pc_error_output_encoding).split('\n')
        data_lines = [x for x in decoded_output if args.pc_error_data_signal in x]
        parsed_data_lines = [x[len(args.pc_error_data_signal):] for x in data_lines]
        # Before last value : information about the metric
        # Last value : metric value
        data = [(','.join(x[:-1]), x[-1]) for x in [x.split(',') for x in parsed_data_lines]]

        if len(headers) == 0:
            headers = additional_headers + [x[0] for x in data]
        compressed_size_in_bits = os.stat(cf).st_size * 8
        n_points_ori = len(PyntCloud.from_file(f).points)
        bpov = compressed_size_in_bits / n_points_ori
        additional_results_row = [f, cf, df, compressed_size_in_bits, n_points_ori, bpov]
        results_row = [float(x[1]) for x in data]

        final_row = additional_results_row + results_row

        assert len(headers) == len(final_row), "Inconsistent output from pc_error"
        results.append(final_row)

    df_results = pd.DataFrame(data=results, columns=headers)
    df_results.to_csv(args.output_file)
 
