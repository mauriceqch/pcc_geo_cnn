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
import itertools
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib as mpl
import metrics

rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
rcParams['legend.fontsize'] = 20
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.figsize'] = 7.3, 4.2

################################################################################
### Script
################################################################################
LINE_STYLES = itertools.cycle(('-', '--', '-.', ':'))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='plot_results.py',
        description='Plot results from list of CSVs for MVUB dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-i','--input', nargs='+', help='Required. List of input csvs.', required=True)
    parser.add_argument('-t','--titles', nargs='+', help='Required. List of method names.', required=True)
    parser.add_argument('output', help='Output file prefix.')

    args = parser.parse_args()

    assert(len(args.input) == len(args.titles))

    logger.info("Checking input files")
    for x in args.input:
        assert os.path.exists(x)
    input_files = args.input
    input_pd = [pd.read_csv(x) for x in input_files]

    seqs = ['andrew', 'david', 'phil', 'ricardo', 'sarah']
    prefixes = [os.path.commonprefix(list(x['ori_file'].values)) for x in input_pd]
    for x, p in zip(input_pd, prefixes):
        x['base_filepath'] = x['ori_file'].map(lambda y: y[len(p):])

    line_styles = [next(LINE_STYLES) for x in args.titles]
    for s in seqs:
        masks = [x['base_filepath'].map(lambda x: x.startswith(s)).values for x in input_pd]
        fig, ax = plt.subplots()
        rd_points = [x.iloc[m][['csv_file', 'bpov', 'Symmetric,rmsFPSNR,p2plane']] for x, m in zip(input_pd, masks)]
        points_per_csv = [x.groupby('csv_file').mean() for x in rd_points]
        points_for_bdrate = [[tuple(y) for y in x.values] for x in points_per_csv]
        bdrate = metrics.bdrate(points_for_bdrate[1], points_for_bdrate[0])
        logger.info(f'bdrate {s} : {bdrate}')
        for points, title, ls in zip(points_per_csv, args.titles, line_styles):
            points.plot(x='bpov', y='Symmetric,rmsFPSNR,p2plane', ax=ax, label=title, style=f'.{ls}')
            ax.set(xlabel='bits per occupied voxel', ylabel='PSNR (dB)')
            ax.set_xlim([0, 2])
            ax.set_ylim([0, 35])
            ax.legend(loc='lower right')
            ax.locator_params(axis='x', nbins=5)

        fig.tight_layout()
        fig.savefig(f'{args.output}_{s}.eps')

    # BDRATE computation
    p1 = input_pd[0].groupby('csv_file').mean()[['bpov', 'Symmetric,rmsFPSNR,p2plane']].values
    p1 = [tuple(x) for x in p1]
    p2 = input_pd[1].groupby('csv_file').mean()[['bpov', 'Symmetric,rmsFPSNR,p2plane']].values
    p2 = [tuple(x) for x in p2]
    bdrate = metrics.bdrate(p2, p1)
    logger.info(f'bdrate: {bdrate}')
    fig, ax = plt.subplots()
    ax.plot([x[0] for x in p1], [x[1] for x in p1], label='Proposed', linestyle='--')
    ax.plot([x[0] for x in p2], [x[1] for x in p2], label='Anchor', linestyle='-.')
    ax.set(xlabel='bits per occupied voxel', ylabel='PSNR (dB)')
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 35])
    ax.legend(loc='lower right')
    ax.locator_params(axis='x', nbins=5)
    fig.tight_layout()
    fig.savefig(f'{args.output}_overall.eps')
