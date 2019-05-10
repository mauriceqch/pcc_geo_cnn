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
import open3d as o3d
################################################################################
### Script
################################################################################
# Value defined in Open3D
# https://github.com/intel-isl/Open3D/blob/master/src/Open3D/Visualization/Visualizer/ViewControl.cpp
ROTATION_RADIAN_PER_PIXEL = 0.003

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='pc_to_camera_params.py',
        description='Generates camera parameters for a point cloud.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'input_path',
        help='Input point cloud path (ply).')
    parser.add_argument(
        'output_path',
        help='Output camera params path.')
    args = parser.parse_args()

    pcd = o3d.read_point_cloud(args.input_path)

    vis = o3d.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    o3d.write_pinhole_camera_parameters(args.output_path, camera_params)
    vis.destroy_window()