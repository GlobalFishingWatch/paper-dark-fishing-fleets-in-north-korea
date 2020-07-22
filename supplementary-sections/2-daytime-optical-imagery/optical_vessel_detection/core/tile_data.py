"""Wrapper For Train / Test Images

"""
import numpy as np
import os
import glob
import skimage.io as skio
import skimage
from skimage import draw
import subprocess
import logging
import tensorflow as tf
import multiprocessing


def RunCustomCommand(command_list):
    print('Running command: %s' % command_list)
    p = subprocess.Popen(command_list,
                         stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    # Can use communicate(input='y\n'.encode()) if the command run requires
    # some confirmation.
    stdout_data, _ = p.communicate()
    print('Command output: %s' % stdout_data)
    logging.info('Log command output: %s', stdout_data)
    if p.returncode != 0:
        raise RuntimeError('Command %s failed: exit code: %s' %
                            (command_list, p.returncode))


def one_in(n, rand=np.random):
    return rand.randint(n) == 0

def path2lbl(x):
    name = os.path.basename(x)
    raw_lbl, size, angle, _ = name.split('__', 3)
    return raw_lbl, float(size), float(angle)

def paths_to_labels(paths):
    return np.array([path2lbl(x) for x in paths])



class TileData(object):

    _weights = None
    _offsets = None
    _delta = None

    def __init__(self, paths, field_of_view, tile_size, resolution, approx_mean, approx_std):
        self.paths = np.asarray(paths)
        self.field_of_view = field_of_view 
        self.tile_size = tile_size
        self.resolution = resolution
        self.approx_mean = approx_mean
        self.approx_std = approx_std

    @property
    def ship_upsample(self):
        raise NotImplementedError()
    
    def __len__(self):
        return len(self.paths)

    def rgba_for(self, path):
        # Actual tiles are tile_size + 1 in size, but 
        # centers are off by 0.5 pixels from center
        # we trim image by one so that centers are exactly at center.
        return skimage.img_as_float(skio.imread(path)[1:, 1:])

    def label_for(self, path):
        return path2lbl(path)

    def _set_weights_offsets_and_delta(self):
        shift_pixels = self.tile_size - self.field_of_view
        assert shift_pixels >= 0
        assert shift_pixels % 2 == 0
        delta = shift_pixels // 2
        raw_offsets = np.arange(-delta, shift_pixels - delta + 1)
        offsets = np.zeros([len(raw_offsets), len(raw_offsets), 2], dtype=int)
        offsets[:, :, 0] = raw_offsets[:, None] 
        offsets[:, :, 1] = raw_offsets[None, :] 
        offsets = offsets.reshape(-1, 2)
        weights = np.ones(len(offsets), dtype=np.float32)
        weights /= weights.sum()
        self._weights = weights
        self._offsets = offsets
        self._delta = delta

    @property
    def weights(self):
        if self._weights is None:
            self._set_weights_offsets_and_delta()
        return self._weights
    
    @property
    def offsets(self):
        if self._offsets is None:
            self._set_weights_offsets_and_delta()
        return self._offsets

    @property
    def delta(self):
        if self._delta is None:
            self._set_weights_offsets_and_delta()
        return self._delta

    def make_mask(self, img, is_ship, size, angle, want_ship):
        offset = self.field_of_view // 2
        nr = self.tile_size - self.field_of_view + 1
        mask = (img[:, :, 3] > 0.5)[offset:offset+nr, offset:offset+nr]
        masked_weights = np.ones(mask.shape, dtype=np.float32)
        masked_weights[~mask] = 0
        ship_mask = np.zeros_like(mask)
        if is_ship:
            c_radius = 0.5 * size
            r_radius = 0.125 * size
            if not want_ship:
                # Add a don't care buffer
                r_radius += 2 * self.resolution
                c_radius += 2 * self.resolution
            rr, cc = draw.ellipse(nr // 2, nr // 2, 
                                  r_radius=r_radius, c_radius=c_radius, 
                                  shape=mask.shape, rotation=angle)
            ship_mask[rr, cc] = 1
        if want_ship:
            masked_weights[~ship_mask] = 0
        else:
            masked_weights[ship_mask] = 0
        return masked_weights

    def choose_random_offsets(self, img, is_ship, size, angle, rand):
        want_ship = one_in(2, rand) and is_ship
        mask = self.make_mask(img, is_ship, size, angle, want_ship)
        masked_weights = mask.reshape(-1)
        assert len(masked_weights) == len(self.offsets)
        if masked_weights.sum():
            masked_weights /= masked_weights.sum()
            ndx = rand.choice(np.arange(len(self.offsets)),
                                   p=masked_weights)
            r0, c0 = self.offsets[ndx]
            is_ship = want_ship
        else:
            logging.warning("Found no good patch for {}, {}, {}".format(is_ship, size, angle))
            r0, c0 = 0, 0
        return r0, c0, is_ship

    def flip_and_transpose(self, img, angle, rand):
        sin = np.sin(angle)
        cos = np.cos(angle)
        if one_in(2, rand):
            sin = -sin
            img = img[::-1, :]
        if one_in(2, rand):
            cos = -cos
            img = img[:, ::-1]
        if one_in(2, rand):
            sin, cos = cos, sin
            img = img.transpose(1, 0, 2) 
        angle = np.arctan2(sin, cos)
        return img, angle

    def add_noise(self, img, rand):
        # by default do nothing
        return img

    def xform_path(self, seed, path):
        if seed == -1:
            seed = None
        rand = np.random.RandomState(seed)
        img = self.rgba_for(path)
        label, size, degrees = self.label_for(path)
        size = float(size)
        angle = np.radians(float(degrees))
        is_ship = (label in ('1', 'E'))
        is_artifact = (label == 'A')

        img, angle = self.flip_and_transpose(img, angle, rand)
        r0, c0, is_ship = self.choose_random_offsets(img, is_ship, size, angle, rand)

        fov = self.field_of_view
        delta = self.delta
        new_img = img[delta+r0:delta+r0+fov, delta+c0:delta+c0+fov, :3].astype(np.float32)
        new_img = (new_img - self.approx_mean) / self.approx_std
        if seed is None:
            # Don't add noise when testing
            new_img = self.add_noise(new_img, rand)
        new_lbls = np.array([[(is_ship, r0, c0, size, angle)]], dtype=np.float32)
        return new_img, new_lbls

    def path_generator(self):
        raise NotImplementedError()

    def as_dataset(self):
        file_ds = tf.data.Dataset.from_generator(self.path_generator, (tf.int32, tf.string))

        def xform(seed, path):
            return tf.py_func(self.xform_path,
                              [seed, path],
                              [tf.float32, tf.float32])
        def set_shapes(X, Y_):
            X.set_shape([self.field_of_view, self.field_of_view, 3])
            Y_.set_shape([1, 1, 5])
            return X, Y_

        def dictify(X, Y_):
            return {'X' : X, 'scene_id' : 'NA'}, Y_

        return (file_ds
                .map(xform, num_parallel_calls=multiprocessing.cpu_count())
                .map(set_shapes) 
                .map(dictify)
                )



class TestTileData(TileData):

    @property
    def ship_upsample(self):
        return 1.0

    def path_generator(self):
        for i, p in enumerate(self.paths):
            yield (i, p) 
        for j, p in enumerate(self.paths):
            yield (i + j, p)

class TrainTileData(TileData):

    _img_weights = None
    _ship_upsample = None

    def _set_img_weights_and_upsample(self):
        is_ship = np.array([(self.label_for(x)[0] in ('1', 'E')) for x in self.paths], 
                            dtype=bool)
        n_ships = is_ship.sum()
        n_empty = len(is_ship) - n_ships
        weights = np.zeros([len(is_ship)], dtype=float)
        # These weights give us positives 1/3 of the time.
        weights[is_ship] = n_empty 
        weights[~is_ship] = n_ships
        # The proportion of imgs with ships presented to the
        # net is 0.5 / 2 = 1/4 when in training mode
        # In test mode it's 0.5 * n_ships / (n_ships + n_empty) == F / 2
        # If we want to weight train time to match that we set
        # weights of ships to W, where 0.25 * W / (0.75 + 0.25 * W) = W / (3 + W) == F / 2
        # Or 2 * W = 3 * F + F * W => W = 3 * F / (2 - F) = 3 * n_ships / (n_ships + 2 * n_empty)
        # So ship upsample isn't quite right, but probably harmless. (For large n_empty/n_ships correct answer is ~2/3 of current)
        self._ship_upsample = n_empty / float(n_ships)
        weights /= weights.sum()
        self._img_weights = weights

    @property
    def img_weights(self):
        if self._img_weights is None:
            self._set_img_weights_and_upsample()
        return self._img_weights

    @property
    def ship_upsample(self):
        if self._ship_upsample is None:
            self._set_img_weights_and_upsample()
        return self._ship_upsample

    def path_generator(self):
        while True:
            yield (-1, np.random.choice(self.paths, p=self.img_weights))





def path_to_base_tile(tile_path):
    """Treat rotated and straight tiles the same"""
    return os.path.splitext(os.path.basename(tile_path))[0].rstrip('R').split('__')[3]

def path_to_id(tile_path):
    return path_to_base_tile(tile_path).rsplit('_', 1)[0].rsplit('_', 2)[0]

def is_train_by_scene(tile_path, split, folds):
    return hash(path_to_id(tile_path)) % folds != split

def is_train_by_tile(tile_path, split, folds):
    return hash(path_to_base_tile(tile_path)) % folds != split



DEFAULT_SPLIT_INFO = {'split' : 0, 'split_type' : 'by-scene'}

class TrainTestTileDataBase(object):

    # Override in subclass
    gcs_tile_location = None
    train_test_folds = None
    tile_size = None
    resolution = None
    approx_mean = None
    approx_std = None
    image_dir = None

    TrainClass = TrainTileData
    TestClass = TestTileData

    def __init__(self, field_of_view, split_info=None):
        if not os.path.exists(self.image_dir):
            self.install()
        split_info = DEFAULT_SPLIT_INFO if (split_info is None) else split_info

        train_paths = self.get_train_paths(split_info)
        logging.info("TILE: %s training tiles", len(train_paths))
        test_paths = self.get_test_paths(split_info)
        logging.info("TILE: %s test tiles", len(test_paths))
        np.random.seed(88)
        np.random.shuffle(train_paths)
        np.random.shuffle(test_paths)
        args = dict(field_of_view=field_of_view, 
                    tile_size=self.tile_size, 
                    resolution=self.resolution, 
                    approx_mean=self.approx_mean, 
                    approx_std=self.approx_std)
        self.train = self.TrainClass(train_paths, **args)
        self.test = self.TestClass(test_paths, **args)

    @classmethod
    def install(cls):
        logging.info('TILES: installing to %s', cls.image_dir)
        RunCustomCommand(['mkdir', '-p', cls.image_dir])
        RunCustomCommand(['gsutil', '-m', 'cp', cls.gcs_tile_location, cls.image_dir])

    @classmethod
    def get_paths(cls):
        image_glob = os.path.join(cls.image_dir, "*.png")
        return sorted([x for x in glob.glob(image_glob)])

    @classmethod
    def get_is_train(cls, split_info):
        split = split_info['split']
        assert split in (-1, 0, 1, 2, 3, 4)
        split = max(split, 0)
        raw_is_train = {
            'by-scene' : is_train_by_scene,
            'by-tile' : is_train_by_tile,
        }[split_info['split_type']]
        return lambda x: raw_is_train(x, split, cls.train_test_folds)

    @classmethod
    def get_train_paths(cls, split_info):
        if split_info['split'] == -1:
            return cls.get_paths() 
        else:
            is_train = cls.get_is_train(split_info)
            return [x for x in cls.get_paths() if is_train(x)]

    @classmethod
    def get_test_paths(cls, split_info):
        is_train = cls.get_is_train(split_info)
        return [x for x in cls.get_paths() if not is_train(x)]  
