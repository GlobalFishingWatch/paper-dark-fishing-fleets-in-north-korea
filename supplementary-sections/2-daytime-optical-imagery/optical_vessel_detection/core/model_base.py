from __future__ import print_function
from __future__ import division
from collections import namedtuple
import numpy as np
import os
import skimage
from skimage import draw
import skimage.util as skutils
import subprocess
import sys
import tensorflow as tf

from . import planet_utils as putils
from .ais_utils import load_ais_info, overlay_ais_on_sceneinfo

WorldDetection = namedtuple("WorldDetection", ["lon", "lat", "size_m", "angle_deg"])
"""A detection represented in world coordinates.
"""

class Detection(namedtuple("Detection", ['r', 'c', 'size', 'angle'])):
    """A detection represented in image coordinates.
    """

    def to_world(self, scene_info):
        """Create transform from (lon, lat) to (column, row) coordinates

        Parameters
        ----------
        scene_info : SceneInfo 
        
        Returns
        ------_
        WorldDetection object that corresponds to self.

    """
        lon, lat = scene_info.cr_to_lonlat(self.c, self.r)
        sz_m = scene_info.pixels_to_m(self.size)
        angle = scene_info.angle_to_world(self.angle)
        return WorldDetection(lon, lat, sz_m, angle)



this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.basename(this_dir)
grandparent_dir = os.path.basename(parent_dir)


# Set the tf random_seed to make this (maybe?) repeatable
# when run on a single processor.
tf.set_random_seed(0.0)

def L1s(x, scale=1.0, width=1.0):
    """Smooth L1 Error

    Like L1 loss except that the region [-0.5, 0.5] is
    has quadratic loss. (This applies for scale=1, 
    width=1, for other values, the region varies).
    The resultant loss and it's derivative are smooth.

    Parameters
    ----------
    x : tf.Tensor
    scale : float
        `x` is scaled down by scale before applying L1s,
        then the resulting loss is scaled back up.
    width : float
        Experimental way to make the quadratic portion
        of the loss curve wider. Making it narrower
        (`width` < 1) does not result in a sensible loss
        and is not allowed. 
    
    Returns
    ------_
    WorldDetection object that corresponds to self.

    """
    if width < 1:
        raise ValueError("width should be >= 1")
    sx = abs(x) / scale
    alpha = width ** (-2 * width)
    beta = 2 * width
    mask = tf.cast(sx < width, tf.float32)
    return scale * (mask * alpha / 2.0 *  sx ** beta + 
                    (1 - mask) * (sx - width + 0.5))



class ModelBase(object):
    """Base class for producing models

    Derived classes should override the parameters below as appropriate
    and implement `model_fn`

    Parameters
    ----------
    model_dir : str
        Directory where model weights and other information are stored.
    split_info : dict
        information on which split to use while training. See
        `tile_data.py` for details.

    """

    # Model parameters
    field_of_view = 213
    scale = 16

    # Training parameters
    steps = 15000
    lr_max_per_batch_size = 0.000125
    lr_decay_steps = 4000
    lr_decay = 0.5
    batch_size = 48
    char_weight = 1
    l2 = 1e-9

    # Training data
    train_data_update_freq = 20
    test_data_update_freq = 100

    # Default model checkpoints loc
    model_dir = os.path.join(grandparent_dir, "checkpoints")

    # Data source (TrainTestTileData for example)
    data_source =  None

    _data = None
    _inference_estimator = None

    def __init__(self, model_dir=None, split_info=None):
        if model_dir is not None:
            self.model_dir = model_dir
        if not model_dir.startswith('gs://'):
            subprocess.check_call(['mkdir', '-p', self.model_dir])

        self.split_info = split_info

    @property
    def lr_max(self):
        """float : learning rate used at the start of training, before decaying it.

        Learning rate varies with batch size, so this keeps the effective learning rate
        constant across batch sizes.
        """
        return self.lr_max_per_batch_size * self.batch_size

    @property
    def name(self):
        """str : name of the subclass to use where we need names."""
        class_module = sys.modules[self.__class__.__module__]
        return os.path.splitext(os.path.basename(class_module.__file__))[0]

    @property
    def data(self):
        """TrainTestTileData : source for training."""
        if self._data is None:
            self._data = self.data_source(field_of_view=self.field_of_view,
                                           split_info=self.split_info)
        return self._data

    @property
    def model_fn(self):
        """function : model function to use in creating an Estimator

        See: https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator
        """
        raise NotImplementedError()
    
    @property
    def inference_estimator(self):
        if self._inference_estimator is None:
            self._inference_estimator = self.make_estimator(None)
        return self._inference_estimator
    

    def make_estimator(self, size, warm_start_from=None, initial_step=0, train_data=None):
        """Create an Estimator object for this model

        Parameters
        ----------
        size : int
            Size of the input field. Typically `self.field_of_view` for
            training or `None` for inference.
        warm_start_from : str or None
            Optional location to pull initial weights for the model from.
        initial_step : int
            Optional training step to start from. Changes the learning rate, etc
            because of learning rate decay.
        train_data : TrainTestTileData
            Optional Training data source. Used to set model training parameters
            such as `ship_upsample`.

        Returns
        -------
        tf.estimator.Estimator
        """
        params={
            'size' : size,
            'char_weight' : self.char_weight,
            'l2' : self.l2,
            'lr_max' : self.lr_max,
            'lr_decay_steps' : self.lr_decay_steps,
            'lr_decay' : self.lr_decay,
            'field_of_view' : self.field_of_view,
            'initial_step' : initial_step
            }
        if train_data:
            params['ship_upsample'] = train_data.ship_upsample



        return tf.estimator.Estimator(
            config=tf.estimator.RunConfig(
                            model_dir=self.model_dir, 
                            save_summary_steps=20,
                            save_checkpoints_secs=300, 
                            keep_checkpoint_max=4,
                    ),
            model_fn=self.model_fn,
            params=params,
            warm_start_from=warm_start_from
        )


    def train_input_fn(self):
        """tf.data.Dataset : Dataset to use for training."""
        return (self.data.train.as_dataset()
                    .prefetch(self.batch_size)
                    .batch(self.batch_size)
                )

    
    def test_input_fn(self):
        """tf.data.Dataset : Dataset to use for validation."""
        return (self.data.test.as_dataset()
                    .prefetch(self.batch_size)
                    .batch(self.batch_size)
                )


    def run_training(self, warm_start_from=None, initial_step=0):
        """train the model

        Parameters
        ----------
        warm_start_from : str or None
            Optional location to pull initial weights for the model from.
        initial_step : int
            Optional training step to start from. Changes the learning rate, etc
            because of learning rate decay.
        """
        train_spec = tf.estimator.TrainSpec(
                        input_fn=self.train_input_fn, 
                        max_steps=self.steps - initial_step)
        eval_spec = tf.estimator.EvalSpec(
                        input_fn=self.test_input_fn,
                        start_delay_secs=120,
                        throttle_secs=300)
        estimator = self.make_estimator(size=self.field_of_view,
                                        warm_start_from=warm_start_from,
                                        initial_step=initial_step,
                                        train_data=self.data.train)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def process_scenes_to_raw(self, scene_infos, batch_size=1):
        """Run inference on scenes

        Parameters
        ----------
        scene_infos: iterable of SceneInfo
        batch_size: int

        Yields
        ------
        dictionary
            raw results for that scene
        """
        pad = self.field_of_view // 2
        prefetch_size = max(batch_size, 4)
        scene_info_map = {}

        def input_fn():

            def generator():
                for scene_info in scene_infos:
                    numbered_scene_id = '{}--{}'.format(scene_info.scene_id, i)
                    scene_info_map[numbered_scene_id] = scene_info
                    # Pad to original size + field_of_view to
                    # avoid worrying about transforms
                    # Pull mean / std from the data_source to avoid installing tiles
                    x = skimage.img_as_float(scene_info.scene[:, :, :3])
                    # Pull mean / std from the data_soure for normalization to avoid installing tiles
                    x = (x - self.data_source.approx_mean) / self.data_source.approx_std
                    # Pad so that final image is same size as original and so that we fill out
                    # to even multiple of scale.
                    h0, w0, _ = x.shape
                    # The dimensions should be of the form 2 * pad + n * scale + 1
                    extra_h = (h0 - 1) % self.scale
                    extra_w = (w0 - 1) % self.scale
                    p_h = (self.scale - extra_h) % self.scale
                    p_w = (self.scale - extra_w) % self.scale
                    x = skutils.pad(x, [(pad, pad + p_h), (pad, pad + p_wd), (0, 0)], 'reflect') 
                    yield {'X' : x, 'scene_id' : numbered_scene_id}
            value_ds = tf.data.Dataset.from_generator(generator, 
                                            {'X' : tf.float32, 'scene_id' : tf.string}) 
            def set_shapes(x):
                x['X'].set_shape([None, None, 3])
                return x
            shaped_ds = value_ds.map(set_shapes).prefetch(prefetch_size) 
            batched_ds = shaped_ds.batch(batch_size)
            return batched_ds

        estimator = self.inference_estimator
        for raw in estimator.predict(input_fn=input_fn):
            scene_info = scene_info.pop(raw['scene_id'])
            yield scene_info, raw

 
    def process_scenes(self, scene_infos, threshold, batch_size=1):
        """Run inference on and postprocess scenes

        Parameters
        ----------
        scene_infos: iterable of SceneInfo
        threshold: float
            Threshold to use for treating each prediction as candidate vessel.
        batch_size: int

        Returns
        -------
        SceneInfo
            original scene info for each scene
        dict
            raw output of the net
        list of Detections        
        """
        for scene_info, raw in self.process_scenes_to_raw(scene_infos, batch_size):
                detections = self.extract_detections(scene_info, raw, threshold)
                yield scene_info, raw, detections


    def extract_detections(self, scene_info, raw, threshold):
        """Extract detections from raw net output

        Parameters
        ----------
        scene_info: SceneInfo
        raw: dict
            raw output of net
        threshold: float
            Threshold to use for treating each prediction as candidate vessel.

        Yields
        ------
        Detection

        """
        scene = scene_info.scene
        Y = raw['is_ship']
        dr0 = raw['dr']
        dc0 = raw['dc']
        size = raw['length_px']
        angle = raw['angle_deg']
        scale = self.scale
        is_ship = Y > threshold
        nr, nc = is_ship.shape
        rs, cs  = np.where(is_ship)
        args = np.argsort(Y[rs, cs])[::-1]
        nr2, nc2 = scene.shape[:2]
        supressed = np.zeros(scene.shape[:2], dtype=bool)
        proposed = np.zeros(scene.shape[:2], dtype=bool)
        for (r, c) in np.transpose([rs, cs])[args]:
            dr = dr0[r, c]
            dc = dc0[r, c]
            ro = int(round(r * scale - dr))
            co = int(round(c * scale - dc))
            ngl = angle[r, c]
            sz = size[r, c]
            # Non max suppression
            delta = max(sz, 16)
            rr0, cc0 = draw.ellipse(ro, co, 
                int(round(0.3 * delta)), # Ships are often close side by side
                int(round(0.6 * delta)), # But not as close longitudinally
                rotation=ngl)
            mask = [(0 <= ri < nr2 and 0 <= ci < nc2) for (ri, ci) in zip(rr0, cc0)]
            if np.mean(mask) < 0.8:
                # Insist that most of boat pair be one screen
                continue
            rr = [x for (x, v) in zip(rr0, mask) if v]
            cc = [x for (x, v) in zip(cc0, mask) if v]
            proposed[:] = 0
            proposed[rr, cc] = 1
            if (proposed & supressed).sum() / proposed.sum() > 0.10:
                continue
            supressed |= proposed
            yield Detection(ro, co, sz, ngl)


    def highlight_scene(self, scene, detections):
        """Overlay detections on scene

        Parameters
        ----------
        scene : np.array 
            RBB or RGBA array.
        detections : sequence of Detection
        """
        for dtct in detections:
            rr, cc = draw.ellipse_perimeter(dtct.r, dtct.c, 
                                  int(round(0.375 * dtct.size)), 
                                  int(round(0.750 * dtct.size)), 
                                  orientation=-dtct.angle, 
                                  shape=scene.shape)
            scene[rr, cc] = (1, 1, 1, 1)

    def add_ais(self, scene_info, metadata, ais_table):
        """Find and overlay AIS related to scene



        """
        ais_df = load_ais_info(metadata, ais_table=ais_table)
        if ais_df is None:
            return scene_info, None
        return overlay_ais_on_sceneinfo(scene_info, ais_df)

    @classmethod
    def run_from_main(cls):
        """Run the model when invoked using `python MODEL.py`
        """
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--initial-step", type=int, default=0, 
                            help="step (for weight decay, etc) to start training from")
        parser.add_argument("--warm-start-from", 
                            help="alternate path to initially load weights from")
        parser.add_argument("--model-dir", 
                            help="directory to store model info in")
        parser.add_argument("--split-type", default="by-scene",
                            help="'by-scene'|'by-tile';"
                                 "by tile is optimistic but has more coverage")
        parser.add_argument("--split", default=4, type=int,
                            help="0-4, or -1 to train on all data, but use split 0 as (bogus) validation")
        args = parser.parse_args()

        tf.logging.set_verbosity(tf.logging.INFO)
        mdl = cls(model_dir=args.model_dir,
                  split_info={'split_type': args.split_type,
                              'split': args.split}
                )
        mdl.run_training(
                warm_start_from=args.warm_start_from,
                initial_step=args.initial_step,
                )