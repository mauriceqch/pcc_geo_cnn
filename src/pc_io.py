import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import multiprocessing
import functools
from tqdm import tqdm
from pyntcloud import PyntCloud
from glob import glob

class PC:
    def __init__(self, points, p_min, p_max):
        self.points = points
        self.p_max = p_max
        self.p_min = p_min
        self.data = {}

        assert np.all(p_min < p_max), f"p_min <= p_max must be true : p_min {p_min}, p_max {p_max}"
        assert np.all(points < p_max), f"points must be inferior to p_max {p_max}"
        assert np.all(points >= p_min), f"points must be superior to p_min {p_min}"

    def __repr__(self):
        return f"<PC with {self.points.shape[0]} points (p_min: {self.p_min}, p_max: {self.p_max})>"

    def is_empty(self):
        return self.points.shape[0] == 0
    
    def p_mid(self):
        p_min = self.p_min
        p_max = self.p_max
        return p_min + ((p_max - p_min) / 2.)

def df_to_pc(df, p_min, p_max):
    points = df[['x','y','z']].values
    return PC(points, p_min, p_max)

def pa_to_df(points):
    df = pd.DataFrame(data={
            'x': points[:,0],
            'y': points[:,1],
            'z': points[:,2]
            })
    
    return df

def pc_to_df(pc):
    points = pc.points
    return pa_to_df(points)

def load_pc(path, p_min, p_max):
    logger.debug(f"Loading PC {path}")
    pc = PyntCloud.from_file(path)
    ret = df_to_pc(pc.points, p_min, p_max)
    logger.debug(f"Loaded PC {path}")

    return ret

def write_pc(path, pc):
    df = pc_to_df(pc)
    pc2 = PyntCloud(df)
    pc2.to_file(path)

def get_shape_data(resolution):
    bbox_min = 0
    bbox_max = resolution
    p_max = np.array([bbox_max, bbox_max, bbox_max])
    p_min = np.array([bbox_min, bbox_min, bbox_min])
    dense_tensor_shape = np.concatenate([[1], p_max]).astype('int64')

    return p_min, p_max, dense_tensor_shape

def get_files(input_glob):
    return np.array(glob(input_glob, recursive=True))

def load_points(files, p_min, p_max):
    files_len = len(files)

    with multiprocessing.Pool() as p:
        logger.info('Loading PCs into memory (parallel reading)')
        pcs = np.array(list(tqdm(p.imap(functools.partial(load_pc, p_min=p_min, p_max=p_max), files, 32), total=files_len)))
        points = np.array(list(tqdm((pc.points for pc in pcs), total=files_len)))

    return points


