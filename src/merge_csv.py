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
import argparse
import multiprocessing
import pandas as pd
from tqdm import tqdm

################################################################################
### Script
################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='merge_csv.py',
        description='Merges a list of csvs into a single output csv. Adds a column indicating the original filename.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-i','--input', nargs='+', help='Required. List of input csvs.', required=True)
    parser.add_argument('output_csv', help='Output csv file.')

    args = parser.parse_args()

    logger.info("Checking input files")
    for x in args.input:
        assert os.path.exists(x)
    input_files = args.input
    input_pd = [pd.read_csv(x) for x in input_files]

    logger.info("Checking column matches")
    columns = [set(x.columns) for x in input_pd]
    assert all(x == columns[0] for x in columns)

    logger.info("Concatenating data")
    for f, x in zip(input_files, input_pd):
        x['csv_file'] = f
    output = pd.concat(input_pd, ignore_index=True, sort=True)

    logger.info("Checking output length")
    assert len(output) == sum((len(x) for x in input_pd))

    logger.info("Writing result")
    output.to_csv(args.output_csv)
    logger.info("Done")
