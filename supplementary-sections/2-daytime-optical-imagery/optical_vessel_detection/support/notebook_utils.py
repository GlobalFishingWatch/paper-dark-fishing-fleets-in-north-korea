from __future__ import print_function
from __future__ import division
from collections import defaultdict, OrderedDict
from glob import glob
import numpy as np
import os
import pandas as pd
from pandas_gbq.gbq import GenericGBQException
import skimage
import skimage.io as skio
import subprocess
import time

from optical_vessel_detection.core.annotation_utils import get_labeled_props
from optical_vessel_detection.core import img_utils as iutils


this_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(this_dir)
grandparent_dir = os.path.dirname(parent_dir)

# Path transforms

def path2sceneid(path):
    """Convert a path to Planet file to a scene id"""
    return os.path.basename(path).rsplit('_', 2)[0]

def path2datestr(path):
    """Convert a path to Planet file to YYYYMMDD"""
    return os.path.basename(path).split('_', 1)[0]

def path2rawtifpath(path):
    """"Convert a path to a Planet file to a path to the corresponding unstraightened tif file"""
    name = path2sceneid(path)
    tifpaths = glob(os.path.join(parent_dir, '../data/paper/scene_info/{}*.tif'.format(name)))
    if len(tifpaths) != 1:
        raise ValueError('could not find tif for', name)
    return tifpaths[0]


# Scenes

def create_scene_map(table_name):
    """Load scene information from BigQuery."""
    query = """
    SELECT * FROM `{}`
    """.format(table_name)
    scene_df = pd.read_gbq(query, dialect='standard')
    def fix_scene_id(x):
        if x.endswith('_3B'):
            x = x[:-3]
        return x
    scene_df['scene_id'] = [fix_scene_id(x) for x in scene_df.scene_id]
    return {x.scene_id : x for x in scene_df.itertuples()}
    

# Detections

def create_detections_map(table_name):
    """Download detection data from BigQuery

    Parameters
    ----------
    table_name : str

    Returns
    -------
    dict of pd.DataFrame
        Mapping of date strings (YYYYMMDD) to DataFrames containing annotation data.
    """
    query = """
    SELECT * FROM `{}`
    """.format(table_name)
    detections_df = pd.read_gbq(query, dialect='standard')
    def fix_scene_id(x):
        if x.endswith('_3B_Visual'):
            x = x[:-10]
        return x
    detections_df['scene_id'] = [fix_scene_id(x) for x in detections_df.scene_id]

    date_array = np.array([x[:8] for x in detections_df.scene_id])
    detections = OrderedDict()
    for date in sorted(set(date_array)):
        mask = (date_array == date)
        detections[date] = detections_df[mask].copy()
        detections[date]['kind'] = ['pair_trawlers'] * mask.sum()

    # deduplicate
    for date, detections_by_date in detections.items():
        deduped = []
        has_dupes = False
        for scene_id in set(detections_by_date.scene_id):
            consolidated = defaultdict(list)
            candidates = detections_by_date[detections_by_date.scene_id == scene_id]
            available_timestamps = set(candidates.processing_timestamp)
            if len(available_timestamps) != 1:
                has_dupes = True
            last_timestamp = max(available_timestamps)
            deduped.extend([x for x in candidates.itertuples() if x.processing_timestamp == last_timestamp])
            detections[date] = pd.DataFrame.from_records(deduped, columns=deduped[0]._fields)
        if has_dupes:
                print("WARNING", date, "has duplicates")
 
    return detections
                

# Annotations

DEFAULT_PROP_MAP =  {
    'false_positive' : (0, 0, 255),
    'pair_trawlers': (255, 0, 0),
    'single_trawler': (255, 128, 128),
    'other_boat': (255, 128, 0),
}
"""Default mapping of names to colors for interpreting annotated images."""


BAD_SCENES = set([
        '20180929_010750_104e',
        '20180602_013142_0f29',
        '20181008_013024_1008',
        "20181008_013026_1008", 
    ])
"""Set of scenes that cause processing to fail.

It appears that they all kill GDAL
"""


def is_straightened(path):
    """Return True if the image on the path straightened.

    Note
    ----
    This is based solely on the location, no actual checking is done.
    """
    suffix = os.path.splitext(path)[1]
    dirname = os.path.basename(os.path.dirname(path))
    if suffix == '.tif' and dirname == 'random_annotations':
        return False
    return True


def _make_xforms(path, scale):
    base_xform = iutils.cr2lonlat_for_geotif(path)
    base_scene = skimage.img_as_float(skio.imread(path))
    scene, straighten_xform, straighten_degrees = iutils.straighten_image(base_scene, scale)
    def cr_to_lonlat(c, r):
        c, r = straighten_xform * (c, r)
        return base_xform(c, r)[:2]
    straighten_angle = np.pi / 180 * straighten_degrees
    def angle_to_world(angle):
        return angle - straighten_angle
    def pixels_to_m(px):
        return 3 * px
    return cr_to_lonlat, angle_to_world, pixels_to_m


def _make_unstraightened_xforms(path, scale):
    base_xform = iutils.cr2lonlat_for_geotif(path)
    def cr_to_lonlat(c, r):
        return base_xform(c, r)[:2]  
    def angle_to_world(angle):
        return angle
    def pixels_to_m(px):
        return 3 * px
    return cr_to_lonlat, angle_to_world, pixels_to_m


def download_raw_tifs(image_paths, src_path, target=os.path.join(grandparent_dir, 'data/paper/scene_info')):
    """Download raw, unrotated tifs corresponding to the images on paths.

    Parameters
    ----------
    image_paths : list of str
    target : str
        directory to copy files to
    """
    template = os.path.join(src_path, '{}/raw/{}')

    names = []
    gs_info_paths = []
    gs_tif_paths = []
    for x in image_paths:
        name = os.path.splitext(os.path.basename(x))[0]
        datestr = name.split('_')[0]
        info_name =  name + '_metadata.xml'
        if not os.path.exists(os.path.join(target, info_name)):
            gs_info_paths.append(template.format(datestr, info_name))
        tif_name = name + '.tif'
        if not os.path.exists(os.path.join(target, tif_name)):
            gs_tif_paths.append(template.format(datestr, tif_name))
    if gs_info_paths:
        subprocess.check_call(['gsutil', '-m', 'cp', '-n'] + gs_info_paths + [target])
    if gs_tif_paths:
        subprocess.check_call(['gsutil', '-m', 'cp', '-n'] + gs_tif_paths + [target])



def create_annotations_map(paths, src_path, table_name, prop_map=DEFAULT_PROP_MAP):
    """Extract annotations data from a set of annotated images.

    Parameters
    ----------
    paths: list of str
        Paths to the annotated images.
    src_path : str
        Base path of imagery in cloud storage.
    table_name: str
        Name of the BigQuery table used to cache results.
    prop_map: dict
        Map of names to colors for extracting annotations from images.

    Note
    ----
    This pulls data from the table if available, so you need to delete
    the table, or use a new name if you want to reprocess the data.
    Also, scenes in BAD_SCENES are skipped.

    Returns
    -------
    dict of pd.DataFrame
        Mapping of date strings (YYYYMMDD) to DataFrames containing annotation data.
    """
    download_raw_tifs(paths, src_path)

    project_id, table_name = table_name.split(':')
    query = """
    select * from `{}`
    """.format(table_name)
    try:
        annotations_df = pd.read_gbq(query, project_id=project_id, dialect='standard')
    except GenericGBQException as err:
        if err.args[0].startswith('Reason: 404 Not found'):
            annotations_df = None
        else:
            raise

    dates = sorted(set([path2datestr(x) for x in paths]))

    processing_timestamp = time.time()
    annotations_map = {}
    if annotations_df is None:
        processed_scene_ids = set()
    else:
        processed_scene_ids = set(annotations_df.scene_id) 
    for date in dates: 
        print(date)
        a4date = defaultdict(list)
        for p in paths:
            name = os.path.basename(p)
            scene_id = path2sceneid(p)
            if not name.startswith(date):
                continue
            if scene_id in processed_scene_ids:
                continue
            if scene_id in BAD_SCENES:
                print("Skipping", scene_id)
                continue
            print('\tprocessing', name)

            # Add dummy annotation to each scene_id so we don't need to reprocess
            a4date['scene_id'].append(scene_id)
            a4date['annotation_r'].append(0.0)
            a4date['annotation_c'].append(0.0)
            a4date['annotation_type'].append('dummy')
            a4date['world_longitude'].append(0.0)
            a4date['world_latitude'].append(0.0)
            a4date['annotation_length_px'].append(0.0)
            a4date['world_length_m'].append(0.0)
            a4date['annotation_angle_rad'].append(0.0)
            a4date['world_angle_rad'].append(0.0)
            a4date['processing_timestamp'].append(processing_timestamp)

            annotated = skio.imread(p)
            props = list(get_labeled_props(annotated, prop_map, min_area=4))
            count = 0
            raw_tif_path = path2rawtifpath(p)
            if is_straightened(p):
                cr_to_lonlat, angle_to_world, pixels_to_m = _make_xforms(
                        raw_tif_path, scale=8)
            else:
                cr_to_lonlat, angle_to_world, pixels_to_m = _make_unstraightened_xforms(
                        raw_tif_path, scale=8)
            for k, v in props:
                for objprops in v:
                    r, c = objprops.centroid
                    a4date['scene_id'].append(scene_id)
                    a4date['annotation_r'].append(r)
                    a4date['annotation_c'].append(c)
                    lon, lat = cr_to_lonlat(c, r)
                    a4date['annotation_type'].append(k)
                    a4date['world_longitude'].append(lon)
                    a4date['world_latitude'].append(lat)
                    an_length_px = objprops.major_axis_length
                    a4date['annotation_length_px'].append(an_length_px)
                    wd_length_m = pixels_to_m(an_length_px)
                    a4date['world_length_m'].append(wd_length_m)
                    an_angle_rad = objprops.orientation + np.pi / 2
                    a4date['annotation_angle_rad'].append(an_angle_rad)
                    wd_angle_rad = angle_to_world(an_angle_rad)
                    a4date['world_angle_rad'].append(wd_angle_rad)
                    a4date['processing_timestamp'].append(processing_timestamp)
                    count += 1
            if count == 0:
                print(path2sceneid(p), "has no annotations")
            processed_scene_ids.add(scene_id)
        data_for_date = pd.DataFrame(a4date,
                columns = ['scene_id', 'annotation_type', 'processing_timestamp',
                    'world_longitude', 'world_latitude', 'world_angle_rad', 'world_length_m',
                    'annotation_r', 'annotation_c', 'annotation_angle_rad', 'annotation_length_px']
        )
        if len(data_for_date):
            data_for_date.to_gbq(table_name, project_id=project_id, if_exists='append')

        if annotations_df is not None:
            mask = [(x.scene_id[:8] == date) for x in annotations_df.itertuples()]
            data_for_date = pd.concat([data_for_date, annotations_df[mask]])

        annotations_map[date] = data_for_date

    # Check for duplicates
    for date, annotations_by_date in annotations_map.items():
        deduped = []
        for scene_id in set(annotations_by_date.scene_id):
            consolidated = defaultdict(list)
            candidates = annotations_by_date[annotations_by_date.scene_id == scene_id]
            available_timestamps = set(candidates.processing_timestamp)
            if len(available_timestamps) != 1:
                    print("WARNING", scene_id, "has", len(available_timestamps),  "timestamps")
            last_timestamp = max(available_timestamps)
            deduped.extend([x for x in candidates.itertuples() if x.processing_timestamp == last_timestamp])
            annotations_map[date] = pd.DataFrame.from_records(deduped, columns=deduped[0]._fields)
            annotations_map[date]['latitude'] = annotations_map[date].world_latitude
            annotations_map[date]['longitude'] = annotations_map[date].world_longitude
            annotations_map[date]['kind'] = annotations_map[date].annotation_type

    return annotations_map