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
import pandas as pd

################################################################################
### Script
################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='fuse_eval_mpeg.py',
        description='Fusion of eval results with custom MPEG anchor output.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'eval_file',
        help='Evaluation file.')
    parser.add_argument(
        'mpeg_eval_file',
        help='MPEG intra evaluation file.')
    parser.add_argument(
        'output_file',
        help='MPEG intra evaluation file.')

    logger.info("Processing started.")
    args = parser.parse_args()

    assert os.path.exists(args.eval_file), "Eval file not found"
    assert os.path.exists(args.mpeg_eval_file), "MPEG eval file not found"
    
    eval_df = pd.read_csv(args.eval_file)
    mpeg_eval_df = pd.read_csv(args.mpeg_eval_file, delimiter=";")
    ori_file_prefix = os.path.commonprefix(list(eval_df.ori_file))
    eval_df['filename'] = eval_df.ori_file.map(lambda s: s[len(ori_file_prefix):])
    eval_df.set_index("filename", inplace=True)
    mpeg_eval_df.set_index("filename", inplace=True)

    fused_df =  eval_df.join(mpeg_eval_df, on="filename")
    fused_df['bpov'] = (fused_df['byte_count_octree_layer'] * 8 / fused_df['n_points_ori'])

    fused_df.to_csv(args.output_file)
    logger.info("Processing done.")
