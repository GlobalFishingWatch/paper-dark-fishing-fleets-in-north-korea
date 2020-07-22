import os
import numpy as np
from skimage import morphology
from .core.tile_data import TrainTestTileDataBase, TrainTileData, one_in


this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)

def truncated_normal(center, sigma, size=(), rand=np.random):
    # Return truncated normal
    values = np.zeros(size, dtype=float)
    x = rand.normal(center, sigma, size=size)
    mask = (abs(x - center) > 2 * sigma)
    while np.sometrue(mask):
        x[mask] = rand.normal(center, sigma, size=mask.sum())
        mask = (abs(x - center) > 2 * sigma)
    return x


class NoisyTrainTileData(TrainTileData):

    def add_noise(self, img, rand):
        # add global noise
        img += truncated_normal(0, 0.1, size=[1, 1, 3], rand=rand)
        img *= np.exp(truncated_normal(0, 0.1, size=[1, 1, 3]))
        return img

class TrainTestTileData(TrainTestTileDataBase):

    gcs_tile_location = "gs://machine-learning-dev-ttl-120d/image_id/pair_trawl_aug_tiles_768/*"
    image_dir = os.path.join(parent_dir, 'data', 'pair_trawl_tiles_768')
    train_test_folds = 5
    tile_size = 768 - 1
    resolution = 16
    image_dir = image_dir
    approx_mean = np.array([0.2796531 , 0.32786318, 0.31866284], 
                dtype=np.float32).reshape(1, 1, -1)
    approx_std = np.array([0.07549028, 0.06564686, 0.06141812],
                dtype=np.float32).reshape(1, 1, -1)


class NoisyTrainTestTileData(TrainTestTileData):

    TrainClass = NoisyTrainTileData
