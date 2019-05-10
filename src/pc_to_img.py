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
import open3d as o3d
################################################################################
### Script
################################################################################
# Value defined in Open3D
# https://github.com/intel-isl/Open3D/blob/master/src/Open3D/Visualization/Visualizer/ViewControl.cpp
ROTATION_RADIAN_PER_PIXEL = 0.003

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='pc_to_img.py',
        description='Map colors from one PC to another.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input_path',
        help='Input point cloud path (ply).')
    parser.add_argument(
        'output_path',
        help='Output image path.')
    parser.add_argument(
        'camera_params_path',
        help='Camera params path.')
    parser.add_argument(
        '--point_size',
        help='Point size.',
        default=1.0,
        type=float
    )
    args = parser.parse_args()

    pcd = o3d.read_point_cloud(args.input_path)

    vis = o3d.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    camera_params = o3d.read_pinhole_camera_parameters(args.camera_params_path)
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    rot_x = -(3.14 / 180) / ROTATION_RADIAN_PER_PIXEL
    rot_y = -(3.14 / 3) / ROTATION_RADIAN_PER_PIXEL
    ctr.rotate(rot_x, rot_y)
    rdr_opt = vis.get_render_option()
    rdr_opt.point_size = args.point_size
    vis.capture_screen_image(args.output_path, True)
    vis.destroy_window()