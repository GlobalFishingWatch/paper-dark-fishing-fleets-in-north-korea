from __future__ import print_function
from __future__ import division
import numpy as np
from skimage import measure
from skimage import draw
from skimage import io as skio
from skimage import img_as_ubyte
import os
import sys
import warnings


class PropMaps(object):

    pair_trawlers = {
            'is_boat': [255, 0, 0],  # Pair of Trawlers, keep 'is_boat' for now for backward compat.
            'single_trawler' : [255, 128, 128],
            'other_boat' : [255, 255, 0],
            'not_boat': [0, 0, 255], 
            'artifact': (0, 255, 0), 
            'dont_know': [255, 0, 255] 
            }

    is_boat = {
            'is_boat': [255, 0, 0], 
            'not_boat': [0, 0, 255], 
            'encounter': [255, 255, 0],
            'artifact': (0, 255, 0), 
            'dont_know': [255, 0, 255] 
            }

    negatives = {
            'not_boat': [0, 0, 255], 
            'artifact': (0, 255, 0), 
            'dont_know': [255, 0, 255] 
    }


char_label_map = {
    'is_boat' : 1,
    'not_boat' : 0,
    'encounter' : 'E',
    'artifact' : 'A',
    'dont_know' : None
}

def get_labeled_props(img, prop_map, min_area=4):
    """Get labeled properties from an annotated image

    Parameters
    ----------
        img : np.array
            An RGB or RGBA image.
        prop_map : dict
            Mapping of label names to (R, G, B) colors.
        min_area : int
            Minimum connected area to consider as a property.
    
    Yields
    ------
        (str, list of skimage.RegionProperties) 
            Tuples of labels and corresponding lists of skimage.RegionProperties

    Examples
    --------

    >>> prop_map =  {'is_boat': [255, 0, 0], 
    ...              'not_boat': [0, 0, 255], 
    ...              'encounter': [255, 255, 0],
    ...              'artifact': (0, 255, 0), 
    ...              'dont_know': [255, 0, 255] 
    ...             }
    >>> for lbl, props in  annotation_utils.get_labeled_props(img, prop_map, min_area):
    ...     do_something_with(lbl, props)

    """
    for lbl, color in prop_map.items():
        is_match = np.logical_and.reduce(img[:, :, :3] == color, axis=2)
        labels = measure.label(is_match, connectivity=2)
        props = measure.regionprops(labels)
        filtered = [p for p in props if p.area >= min_area]
        if filtered:
            yield lbl, [p for p in props if p.area >= min_area]


def create_tiles_from_annotated_img(raw_img_path, annotated_img_path, tile_size, tile_dir, 
                                    prop_map):
    """Create tiles for training nnet from images annotated with colored lines or patches

    Parameters
    ----------
        raw_img_path : str
            Path to the original, non-annotated image. 
        annotated_img_path : str
            Path to annotated image. This must align with the original image.
            Typically, the original image is just marked up.
        tile_size : int
            Size (N) of the NxN tiles to generate.
        tile_dir : str
            Path to the directory to place generate tiles in.
        prop_map: dict
            Mapping used to interpret annotations. See get_labeled_props

    """
    assert tile_size % 2 == 0
    delta = tile_size // 2

    name = os.path.splitext(os.path.basename(raw_img_path))[0]
    target = skio.imread(annotated_img_path)
    masks = np.zeros_like(target[:, :, 0], dtype=int)
    img = skio.imread(raw_img_path).astype(np.float32) / 255.0
    nr, nc, _ = img.shape
    for long_lbl, props in get_labeled_props(target, prop_map, min_area=4):
        lbl = char_label_map[long_lbl]
        for i, p in enumerate(props):
            r, c = [int(round(x)) for x in p.centroid]
            # We mask stuff on edge even though we don't create tiles.    
            if lbl in (1, None):
                size = p.major_axis_length
                angle = np.degrees(p.orientation + np.pi / 2)
                rr, cc = draw.ellipse(r, c, 0.25 * size, 0.5 * size,
                    rotation=p.orientation + np.pi / 2, shape=masks.shape)
                masks[rr, cc] = True
                masks[rr, cc] = True
            else:
                size = angle = 0
    for long_lbl, props in get_labeled_props(target, prop_map, min_area=4):
        if long_lbl == 'dont_know':
            continue
        lbl = char_label_map[long_lbl]    
        for i, p in enumerate(props):
            r, c = [int(round(x)) for x in p.centroid]
            submasks = masks[r - delta:r + delta, c - delta: c + delta]
            mask = (submasks == 0) 
            if not (delta <= r <= nr - delta):
                continue
            if not (delta <= c <= nc - delta):
                continue     
            if lbl == 1:
                size = p.major_axis_length
                angle = np.degrees(p.orientation + np.pi / 2)
                rr, cc = draw.ellipse(delta, delta, 0.25 * size, 0.5 * size,
                    rotation=p.orientation + np.pi / 2, shape=mask.shape)
                mask[rr, cc] = True
            else:
                rr, cc = draw.circle(delta, delta, 8,
                     shape=mask.shape)
                mask[rr, cc] = True
                size = angle = 0
            tile = img[r - delta:r + delta, c - delta: c + delta]
            depth = tile.shape[-1]
            if depth == 3:
                tile = np.concatenate([tile, 0 * tile[:, :, :1]], axis=2)
            tile[:, :, 3] = mask
            assert tile.shape[:2] == (tile_size, tile_size), (tile_size, tile.shape)
            img_name = '{}__{:.1f}__{:.1f}__{}_{}-{}.png'.format(lbl, size, angle, name, r, c)
            path = os.path.join(tile_dir, img_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skio.imsave(path, img_as_ubyte(tile))
            print(lbl, end=''); sys.stdout.flush()