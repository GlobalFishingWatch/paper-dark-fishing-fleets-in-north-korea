from __future__ import print_function
from __future__ import division
from itertools import tee
import logging
import numpy as np
import tensorflow as tf
import skimage
import skimage.util as skutils
import six
from threading import Lock

from .core.model_base import ModelBase, L1s

class safeteeobject(object):
    """tee object wrapped to make it thread-safe"""
    # From https://stackoverflow.com/questions/6703594/is-the-result-of-itertools-tee-thread-safe-python
    def __init__(self, teeobj, lock):
        self.teeobj = teeobj
        self.lock = lock
    def __iter__(self):
        return self
    def __next__(self):
        with self.lock:
            return next(self.teeobj)
    def next(self):
        return self.__next__()
    def __copy__(self):
        return safeteeobject(self.teeobj.__copy__(), self.lock)

def safetee(iterable, n=2):
    """tuple of n independent thread-safe iterators"""
    lock = Lock()
    return tuple(safeteeobject(teeobj, lock) for teeobj in tee(iterable, n))

class GPUInferenceModel(ModelBase):

    def process_scenes_to_raw(self, scene_infos, batch_size=1):
        """Run inference on scenes

        Parameters
        ----------
        scene_infos: iterable of SceneInfo
        batch_size: int

        Yields
        ------
        SceneInfo
            original scene info for each scene
        dictionary
            raw results for that scene
        """
        pad = self.field_of_view // 2
        prefetch_size = max(batch_size, 4)
        scene_info_map = {}

        def extra_padding(n, extra=1):
            extra = (n - extra) % self.scale
            return (self.scale - extra) % self.scale

        def input_fn():

            MAX_WIDTH = 3520
            assert MAX_WIDTH % self.scale == 0

            def generator():
                for i, scene_info in enumerate(scene_infos):
                    scene_id = scene_info.scene_id
                    numbered_scene_id = '{}--{}'.format(scene_id, i)
                    scene_info_map[numbered_scene_id] = scene_info
                    logging.debug('Creating info subscenes for: {}'.format(scene_id))
                    x = skimage.img_as_float(scene_info.scene[:, :, :3])
                    del scene_info # Free some space
                    # Pull mean / std from the data_soure for normalization to avoid installing tiles
                    x = (x - self.data_source.approx_mean) / self.data_source.approx_std
                    h0, w0, _ = x.shape
                    is_transposed = (h0 > w0)
                    if is_transposed:
                        x = x.transpose(1, 0, 2)
                        h0, w0 = w0, h0
                    logging.debug("initial shape {}".format(x.shape))
                    # Dimensions should be of the form 2 * pad + n * scale + 1
                    p_h = extra_padding(h0)
                    p_w = extra_padding(w0, 3)
                    x = skutils.pad(x, [(pad, pad + p_h), (pad, pad + p_w), (0, 0)], 'reflect') 
                    logging.debug("padded shape {}".format(x.shape))
                    h, w, _ = x.shape
                    # Break each image into three pieces so that we can run on GPU
                    dw = w0 // 3 + extra_padding(w0 // 3)
                    assert (dw - 1) % self.scale == 0, ('all chunks should align with scale', dw, i1, i2)
                    i1 = dw + pad
                    i2 = w - (dw + pad)
                    assert i1      > pad, (h, w, x.shape)
                    assert i2 - i1 > 0,   (h, w, x.shape)
                    assert w  - i2 > pad, (h, w, x.shape)
                    logging.debug("breakpoints {}, {}, {}, {}".format(i1, i2, dw, pad))
                    logging.debug('yielding first part for : {}'.format(scene_id))
                    subx = x[:, :i1 + pad]
                    logging.debug("shape1 {}".format(subx.shape))
                    assert (subx.shape[1] - 1 - 2 * pad) % self.scale == 0, ('all chunks should align with scale', w, dw, i1, i2)
                    yield {'X': subx, 'scene_id' : numbered_scene_id}
                    logging.debug('yielding second part for : {}'.format(scene_id))
                    subx = x[:, i1 - pad + self.scale:i2 + pad - self.scale]
                    logging.debug("shape2 {}".format(subx.shape))
                    assert (subx.shape[1] - 1 - 2 * pad) % self.scale == 0, ('all chunks should align with scale', w, dw, i1, i2)
                    yield {'X': subx,  'scene_id' : numbered_scene_id}
                    logging.debug('yielding third part for : {}'.format(scene_id))
                    subx = x[:, i2 - pad:]
                    logging.debug("shape3 {}".format(subx.shape))
                    assert (subx.shape[1] - 1 - 2 * pad) % self.scale == 0, ('all chunks should align with scale', w, dw, i1, i2)
                    yield {'X': subx,  'scene_id' : numbered_scene_id}
                    logging.debug('finished : {}'.format(scene_id))
            value_ds = tf.data.Dataset.from_generator(generator, 
                                            {'X' : tf.float32, 'scene_id' : tf.string}) 
            def set_shapes(x):
                x['X'].set_shape([None, None, 3])
                return x
            shaped_ds = value_ds.map(set_shapes).prefetch(prefetch_size) 
            batched_ds = shaped_ds.batch(batch_size)
            return batched_ds

        def grouped(iterable):
            group_iter = iter(iterable)
            group_exhausted = False
            while True:
                try:
                    x_a = next(group_iter)
                except StopIteration:
                    logging.debug('data exhausted')
                    return
                else:
                    try:
                        x_b = next(group_iter)
                        x_c = next(group_iter)
                    except StopIteration:
                        raise RuntimeError('data exhausted mid group')
                if not (x_a['scene_id'] == x_b['scene_id'] == x_c['scene_id']):
                    raise RuntimeError('group scene_ids do not match')
                key = six.ensure_text(x_a['scene_id'])
                scene_info = scene_info_map.pop(key)
                yield (scene_info, x_a, x_b, x_c)

        estimator = self.inference_estimator
        for scene_info, x_a, x_b, x_c in grouped(estimator.predict(input_fn=input_fn)): 
            results = {}
            h, w, _ = scene_info.scene.shape
            is_transposed = (h > w)
            if is_transposed:
                h, w = w, h
            for k in ['is_ship', 'dr', 'dc', 'length_px', 'angle_deg']:
                x = np.concatenate([x_a[k], x_b[k], x_c[k]], axis=1)
                assert len(x.shape) == 2
                assert (h + extra_padding(h) - 1) / self.scale + 1 == x.shape[0], (h, x.shape, is_transposed)
                assert (w + extra_padding(w, 3) - 3) / self.scale + 1 == x.shape[1], (w, x.shape, is_transposed)
                if is_transposed:
                    x = x.transpose(1, 0)
                results[k] = x
            yield scene_info, results



if __name__ == '__main__':
    Model.run_from_main()
