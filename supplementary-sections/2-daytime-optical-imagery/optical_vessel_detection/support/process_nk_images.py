from __future__ import print_function
from __future__ import division
import matplotlib.path 
matplotlib.use('Agg')
# export PL_API_KEY=...
from collections import namedtuple
import datetime
import dateutil
import fnmatch
from glob import glob
from google.cloud import storage
import json
import logging
import matplotlib.path as mplpath
import numpy as np
import os
import pandas as pd
import shutil
import skimage
import skimage.io as skio
import sys
import tempfile
import time
import uuid
import warnings

from optical_vessel_detection.core import img_utils as iutils
from optical_vessel_detection.core.img_utils import SceneInfo
from optical_vessel_detection.core import planet_utils as putils
from optical_vessel_detection.support import regions

this_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(this_dir)

proj_id = os.environ['PROJ_ID']

from optical_vessel_detection.pair_trawler_model_v1_1 import GPUInferenceModel as Model

MODEL_DIR = 'gs://machine-learning-production/planet_paper/model_weights/v1_1_aug/'
AIS_INFO_TABLE = None

PathInfo = namedtuple("PathInfo", ["source", "target", "scene_id"])

valid_poly = mplpath.Path(regions.full_study_area)


def is_valid(sinfo):
    """check if a scene intersects `valid_poly`.

    The actual check is whether any corners are inside `valid_poly`, 
    which can occasionally go awry, but is generally good enough.

    Parameters
    ----------
    sinfo: SceneInfo

    Returns
    -------
    bool

    """
    pts = []
    for x in sinfo['boundary']:
        pts.append((x['lon'], x['lat']))
    return np.sometrue(valid_poly.contains_points(pts))


def mkdir_if_needed(path):
    """Make a new directory if needed.

    Ignores paths starting with 'gs://' since GCS automatically
    creates "directories" as needed. Except for new buckets --
    the bucket needs to exist already if using GCS.

    Parameters
    ----------
    path: str
    """
    if path.startswith('gs://'):
        return
    if not os.path.exists(path):
        os.makedirs(path)


def path_source(raw_path, detect_prefix):
    """Generate paths for all available images.

    Generates paths for all images located in the
    directory specified by `raw_path`. Images corresponding
    to existing results in `detect_prefix` are ignored.

    Parameters
    ----------
    raw_path : str
        Path to directory containing a set of `tif` image files
        and `xml` metadata files.
    detect_prefix : str
        Path to directory containing detections in `json` files. 

    Yields
    ------
    PathInfo

    """
    paths = GcsHelper.list_paths(os.path.join(raw_path, '*.tif'))
    xml_ids = set([os.path.basename(x).rsplit('_', 1)[0] for x in GcsHelper.list_paths(os.path.join(raw_path, '*.xml'))])
    paths.sort()
    np.random.seed(888)
    np.random.shuffle(paths)
    existing_paths = set(GcsHelper.list_paths(os.path.join(detect_prefix, '*.json')))
    for src in paths:
        logging.debug('considering path: {}'.format(src))
        scene_id = os.path.splitext(os.path.basename(src))[0]
        if scene_id not in xml_ids:
            logging.warning('Skipping path: {} because xml is missing'.format(src))
            continue
        tgt = os.path.join(detect_prefix, scene_id)
        if (tgt + '.json') in existing_paths:
            logging.info('Skipping path: {} because exists'.format(src))
            continue
        logging.info('yielding path: {}'.format(src))
        yield PathInfo(src, tgt, scene_id)


def scene_json_to_df(obj):
    """Convert json describing detections in a scene to a DataFrame

    Parameters
    ----------
    obj : dict
        Json compatible object

    Returns
    -------
    pd.DataFrame

    """
    n = len(obj['detections'])
    data = dict(
    scene_id = [ obj['scene'] ] * n,
    processing_timestamp = [ obj['process_timestamp'] ] * n,
    longitude = [x['lon'] for x in obj['detections']],
    latitude = [x['lat'] for x in obj['detections']],
    vessel_size = [x['size_m'] for x in obj['detections']],
    vessel_angle = [x['angle_deg'] for x in obj['detections']]
    )
    return pd.DataFrame(data,
                       columns=['scene_id', 'processing_timestamp', 'latitude', 'longitude',
                               'vessel_size', 'vessel_angle'])


def make_scene_info(p):
    """Create SceneInfo for PathInfo object

    Parameters
    ----------
    p : PathInfo

    Returns
    -------
    SceneInfo

    """
    with GcsHelper() as gh:
        xml_path = gh.to_local(p.source.replace('.tif', '_metadata.xml'))
        metadata = putils.scene_info_from_path(xml_path)
        if not is_valid(metadata):
            logging.debug('skipping path: {} because invalid'.format(p.source))
            return None
        source = gh.to_local(p.source)
        base_xform = iutils.cr2lonlat_for_geotif(source)
        rev_xform = iutils.lonlat2cr_for_geotif(source)
        base_scene = skimage.img_as_float(skio.imread(source))
        scene, straighten_xform, base_degrees =  iutils.straighten_image(base_scene)
        scene = iutils.fast_fill(scene)
        base_angle = np.pi / 180 * base_degrees
        def cr_to_lonlat(c, r):
            return base_xform(*(straighten_xform * (c, r)))[:2]
        def lonlat_to_cr(lon, lat):
            return ~straighten_xform * rev_xform(lon, lat)
        def angle_to_world(angle):
            return angle - base_angle
        def pixels_to_m(px):
            return 3 * px
        logging.debug('returning SceneInfo for path: {}'.format(p))
        return SceneInfo(p, scene, p.scene_id, cr_to_lonlat, angle_to_world, pixels_to_m,
                        lonlat_to_cr)


def raw2raster(raw):
    return np.transpose([
        raw['is_ship'],
        raw['dr'],
        raw['dc'],
        raw['length_px'],
        raw['angle_deg'],
        ], axes=(1, 2, 0))

def raster2raw(raster):
    return {
        'is_ship' :   raster[:, :, 0],
        'dr' :        raster[:, :, 1],
        'dc' :        raster[:, :, 2],
        'length_px' : raster[:, :, 3],
        'angle_deg' : raster[:, :, 4],
    }


def process_scenes(p_source, min_detections_to_save, detections_table):
    """Run models across scenes.

    Parameters
    ----------
    p_source 
        Iterable that returns PathInfo objects.
    min_detections_to_save : int
        Minimum number of detections needed to trigger saving an annotated
        image. Json and BQ data are always saved.
    detections_table : str
        BQ table to write detections to.
    
    Yields
    ------
    str
        path of original image
    list
        list of detected pair trawlers

    """


    def scene_source():
        for p in p_source:
            # By making scene info in a separate function we avoid potential issues with
            # a race condition
            sinfo = make_scene_info(p)
            if sinfo is not None:
                yield sinfo
    
    model = Model(MODEL_DIR)
    detect_iter =  model.process_scenes(scene_source(),
                                        threshold=0.95)

    for base_scene_info, raw, detections in detect_iter:
        with GcsHelper() as gh:
            xml_path = gh.to_local(base_scene_info.path.source.replace('.tif', '_metadata.xml'))
            metadata = putils.scene_info_from_path(xml_path)

            if AIS_INFO_TABLE:
                scene_info, ais_info = model.add_ais(base_scene_info, metadata, AIS_INFO_TABLE)
            else:
                scene_info, ais_info = base_scene_info, None
            del base_scene_info # Save some memory
            p = scene_info.path
            scene = scene_info.scene
            detections = list(detections)
            
            world_detections = [x.to_world(scene_info) for x in detections]
            
            final_path = p.target + '.json'
            path = gh.get_local_path(suffix='.json')
            base_obj = {
                'scene' : p.scene_id,
                'detections' : [x._asdict() for x in world_detections],
                'process_timestamp' : datetime.datetime.utcnow(),
                }
            json_obj = base_obj.copy()
            json_obj['process_timestamp'] = json_obj['process_timestamp'].isoformat() + 'Z'
            if ais_info is not None:
                json_obj['ais'] = ais_info
            with open(path, 'w') as f:
                f.write(json.dumps(json_obj))
            df = scene_json_to_df(base_obj)
            if df.shape[0] > 0:
                df.to_gbq(detections_table, 
                          proj_id, if_exists='append')
            gh.copy_from_local(path, final_path)

            # Build raw_stack and save as tif
            final_path = p.target + '_raw.tif'
            path = gh.get_local_path(suffix='.tif')
            skio.imsave(path, raw2raster(raw))
            gh.copy_from_local(path, final_path)

            if len(detections) >= min_detections_to_save:  
                model.highlight_scene(scene_info.scene, detections)
                final_path = p.target + '.png'
                path = gh.get_local_path(suffix='.png')
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    skio.imsave(path, scene) 
                gh.copy_from_local(path, final_path)

        yield scene_info.path, detections


def process_images(source, target, min_detections, detections_table):
    """Process the images from a given source directory.

    Parameters
    ----------
    source : str
        Input images are in `source`/raw
    target : str
        Output json and image files are written to `target`/detections
    min_detections : int
        Minimum detections needed to write out an annotated image file
    detections_table : str
        BQ table to write detections to.


    Returns
    ------
    int
        Number of images that will be processed
    generator
        Yields (scene_id, detections)

    Note
    ----
        Number of images actually processed may be slightly less than
        advertised due to is_valid checks.
    """
    raw_path = os.path.join(source, 'raw')
    detect_path = os.path.join(target, 'detections')

    mkdir_if_needed(detect_path)

    p_source = list(path_source(raw_path, detect_path))

    # Supress unneeded TF warning about memory
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

    def processor():
        for p, detections in process_scenes(p_source, min_detections, detections_table):
            yield (p.scene_id, detections)

    return len(p_source), processor()


class GcsHelper(object):
    """Wrapper function to access common gsutil functions

    Most methods are static and thus no class instance needs to be
    instantiated. For example to list paths in a local or gcs directory"

        GcsHelper.list_paths('path_to_local_dir') 
        GcsHelper.list_paths('gs://path_to_gcs_dir') 

    However, `get_local_path` and `to_local` allocate
    temporary local storage and thus should only be used in as `with`
    statement. For example:

        with GcsHelper() as gh:
            local_path = gh.to_local('gs://path_to_gcs_object')
            # do something using local_path
        # the object downloaded to local_path is deleted.



    """
    def __init__(self):
        self.temp_dir = None
        self.cache = None

    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache = {}
        return self

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self.temp_dir)
        self.temp_dir = None
        self.cache = None

    def get_local_path(self, suffix='.tmp'):
        if self.temp_dir is None:
            raise ValueError('can only be used with a `with` statement')
        name = '{}{}'.format(uuid.uuid1().hex, suffix)
        return os.path.join(self.temp_dir, name)

    @staticmethod
    def _list_blobs(bucket_name, prefix):
        """Lists all the blobs in the bucket."""
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        return bucket.list_blobs(prefix=prefix)

    @staticmethod
    def _download_blob(bucket_name, source_blob_name, destination_file_name):
        """Downloads a blob from the bucket."""
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
    
    @staticmethod
    def _upload_blob(source_file_name, bucket_name, destination_blob_name):
        """Uploads a file to the bucket."""
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
    
    @staticmethod
    def list_gcs_paths(pattern):
        assert pattern.startswith('gs://')
        bucket_name, blob_name = pattern[5:].split('/', 1)
        special_chars = '*?[!'
        n = len(blob_name)
        for ch in special_chars:
            try:
               n = min(n, blob_name.index(ch))
            except:
                pass
        prefix = blob_name[:n]
        blobs = GcsHelper._list_blobs(bucket_name, prefix)
        return list([('gs://' +  bucket_name + '/' + x.name) 
                     for x in blobs if fnmatch.fnmatch(x.name, blob_name)])

    @staticmethod
    def list_paths(path):
        if path.startswith('gs://'):
            return GcsHelper.list_gcs_paths(path)
        else:
            if os.path.isdir(path):
                return [os.path.join(path, x) for x in os.path.listdir(path)]
            else:
                return glob(path)

    def to_local(self, path):
        if not path.startswith('gs://'):
            return path
        if path in self.cache:
            return self.cache[path]
        suffix = os.path.splitext(path)[1]
        local_path = self.get_local_path(suffix=suffix)
        bucket_name, blob_name = path[5:].split('/', 1)
        self._download_blob(bucket_name, blob_name, local_path)
        self.cache[path] = local_path
        return local_path

    @staticmethod
    def copy_from_local(local_path, path):
        if path.startswith('gs://'):
            bucket_name, blob_name = path[5:].split('/', 1)
            GcsHelper._upload_blob(local_path, bucket_name, blob_name)
        else:
            shutil.copyfile(local_path, path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="nk_detection",
                        help="source directory to pull images from")
    parser.add_argument("--target", default="nk_detection",
                        help="target directory to store output in")
    parser.add_argument("--date", action="append", dest="dates",
                        help="date to download images for")
    parser.add_argument("--min-detections", type=int, default=3,
                        help="minimum detections to flag")
    parser.add_argument("--poll-interval-minutes", type=int,
                       help="interval to retry download, by default do not retry")
    parser.add_argument("--detections-table", 
                        default="machine_learning_dev_ttl_120d.detected_pair_trawlers",
                        help="BQ table to upload detections to")
    parser.add_argument("--log-level", default='WARNING', choices=['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'], 
                         help="level of logging events to output")
    parser.add_argument("--gpu", help="comma separated list of GPU ids or -1 to force CPU")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu   
    
    while True:
        print("Checking for scenes to process at {}:".format(datetime.datetime.now().isoformat()))
        # Refactor;  used more that just here (download_nk_images at least)
        source = args.source
        target = args.target

        if args.dates:
            timestamps = []
            for x in args.dates:
                ts = dateutil.parser.parse(x)
                assert ts.time() == datetime.time()
                timestamps.append(ts)
        else:
            # By default, process any unprocessed images.
            paths = GcsHelper.list_paths(source)
            timestamps = []
            for x in sorted([os.path.basename(y) for y in paths]):
                try:
                    ts = dateutil.parser.parse(x)
                except: 
                    continue
                timestamps.append(ts)

        for ts in timestamps: 
            datestr = "{:%Y%m%d}".format(ts.date())
            dated_source = os.path.join(source, datestr)
            dated_target = os.path.join(target, datestr)

            unprocessed_img_count, processor = process_images(dated_source, dated_target, 
                    args.min_detections, args.detections_table)
            print("    {:%Y-%m-%d}: {}".format(ts, unprocessed_img_count))
            was_below_threshold = False
            for scene_id, detections in processor:
                below_threshold = (len(detections) < args.min_detections)
                if below_threshold:
                    if was_below_threshold != True:
                        print('        ', end='')
                    print('.', end='')
                    sys.stdout.flush()
                else:
                    if was_below_threshold == True:
                        print()
                    print('        ' + scene_id)
                was_below_threshold = below_threshold
            if was_below_threshold:
                print()

        if args.poll_interval_minutes is None:
            break
        else:
            time.sleep(args.poll_interval_minutes * 60)



