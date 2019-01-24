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
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools

################################################################################
### Script
################################################################################
LINE_STYLES = itertools.cycle(('-', '--', '-.', ':'))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='plot_results.py',
        description='Plot results from list of CSVs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-i','--input', nargs='+', help='Required. List of input csvs.', required=True)
    parser.add_argument('-t','--titles', nargs='+', help='Required. List of titles.', required=True)
    parser.add_argument('output', help='Output file.')

    args = parser.parse_args()

    assert(len(args.input) == len(args.titles))

    logger.info("Checking input files")
    for x in args.input:
        assert os.path.exists(x)
    input_files = args.input
    input_pd = [pd.read_csv(x) for x in input_files]

    fig, ax = plt.subplots()
    rd_points = [x[['csv_file', 'bpov', 'Symmetric,rmsFPSNR,p2plane']] for x in input_pd]
    points_per_csv = [x.groupby('csv_file').mean() for x in rd_points]
    for points, title, ls in zip(points_per_csv, args.titles, LINE_STYLES):
        points.plot(x='bpov', y='Symmetric,rmsFPSNR,p2plane', ax=ax, label=title, style=f'.{ls}')
    fig.show()

