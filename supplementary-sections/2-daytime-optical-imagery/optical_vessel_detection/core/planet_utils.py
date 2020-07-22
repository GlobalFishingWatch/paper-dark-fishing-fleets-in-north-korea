from __future__ import print_function
from __future__ import division
import sys
from planet import api
from planet.api import APIException
from glob import glob
import time
import os
from collections import defaultdict
from skimage import io
from skimage import transform
from skimage import measure
import xml.etree.ElementTree as ET
import numpy as np
import sys
import tempfile
import subprocess
import shutil
import time
import affine
import pandas as pd
import json
from . import img_utils as iutils

proj_id = os.environ['PROJ_ID']

class Downloader(object):
    """Download data from Planet given a scene_id

    Basic Usage:

       Downloader().retrieve(list_of_scenes, destination_dir)

    """

    def __init__(self, asset_types=("PSScene3Band", "PSScene4Band")):
        self.client = api.ClientV1()
        self.asset_types = asset_types

    def activate_id(self, scene_id, verbose=False):
        for asset_type in self.asset_types:
            try:
                if verbose:
                    print('attempting to activate:', scene_id, asset_type, end=' ')
                    sys.stdout.flush()
                assets = self.client.get_assets_by_id(asset_type, scene_id).get()
            except api.MissingResource:
                print('failed due to MissingResource for', asset_type)
            else:
                self.client.activate(assets['visual'])
                self.client.activate(assets['visual_xml'])
                if verbose:
                    print('succeeded')
                return (asset_type, scene_id)
        else: 
            raise ValueError("Could not find asset for", repr(scene_id))

    def activate_ids(self, scene_ids, verbose=False):
        for sid in scene_ids:
            try:
                yield self.activate_id(sid, verbose=verbose)
            except APIException as err:
                print("failed to activate {} ({})".format(sid, err))

    def download_assets(self, typed_ids, dest, 
                        tries=20, verbose=False, gcs_delay=30):
        is_gcs_dest = dest.startswith('gs:')
        if is_gcs_dest:
            local_dest = tempfile.mkdtemp()
        else:
            local_dest = dest

        try:
            pending = set(typed_ids)
            for i in range(tries):
                if verbose:
                    print("Try", i)
                for item in sorted(pending):
                    atype, sid = item
                    assets = self.client.get_assets_by_id(atype, sid).get()
                    if (assets['visual']['status'] == 
                            assets['visual_xml']['status'] == 'active'):
                        if verbose:
                            print('.', end=''); sys.stdout.flush()
                        self.client.download(assets['visual'], 
                                        api.utils.write_to_file(local_dest))
                        self.client.download(assets['visual_xml'], 
                                        api.utils.write_to_file(local_dest))
                        pending.remove(item)
                if not pending:
                    break
                time.sleep(30)
            if verbose:
                print()
            if is_gcs_dest:
                # There is some sort of race condition here which causes
                # intermittent failure of the copy, so sleep for a while
                # to allow to complete.
                time.sleep(gcs_delay)
                command = ["gsutil",  "-m",  "cp",  
                                os.path.join(local_dest, '*'), dest]
                if verbose:
                    print("Executing:", ' '.join(command))
                subprocess.check_call(command)
            if verbose:
                print("Complete")
        finally:
            if is_gcs_dest:
                shutil.rmtree(local_dest)


    def retrieve(self, scene_ids, dest, tries=20, verbose=False):
        typed_ids = list(self.activate_ids(scene_ids, verbose=verbose))
        self.download_assets(typed_ids, dest, tries, verbose)



def _find_angle(blank):
    r, c = blank.shape
    for i in range(r):
        if (~blank[i, :]).sum():
            break
    blank = blank[i:]
    for j in range(c):
        if (~blank[:, j]).sum():
            break
    blank = blank[:, j:]
    a = np.searchsorted(np.cumsum(~blank[:, 0]), 1, side='left')
    b = np.searchsorted(np.cumsum(~blank[0, :]), 1, side='left')
    return np.degrees(np.arctan2(b, a))



def find_angle(blank):
    return 0.25 * (
                   _find_angle(blank) +
                   90 - _find_angle(blank[::-1]) +
                   90 - _find_angle(blank[:, ::-1]) +
                   _find_angle(blank[::-1, ::-1])
                )


def rotate_and_clip(img, blank, degrees, aux=None):
    nr, nc = img.shape[:2]
    radians = np.radians(degrees)
    # Rotate image
    img = img.astype(float) / img.max()
    new_img = transform.rotate(img, degrees)
    new_blank = transform.rotate(blank, degrees, order=0, mode='constant', cval=1) > 0.5
    if aux is not None:
        new_aux = transform.rotate(aux, degrees, mode='edge', preserve_range=True)
    # First color in extra with red to show clip region
    valid = ~new_blank
    for i0 in range(nr):

        if valid[i0].sum():
            break
    for i1 in range(nr - 1, -1, -1):
        if valid[i1].sum():
            break
    for j0 in range(nc):
        if valid[:, j0].sum():
            break
    for j1 in range(nc - 1, -1, -1):
        if valid[:, j1].sum():
            break
    if aux is not None:
        return new_img[i0:i1, j0:j1], new_aux[i0:i1, j0:j1]
    else:
        center = np.array((nc, nr)) / 2. - 0.5
        trans = affine.Affine.translation
        xform = trans(*center) * affine.Affine.rotation(degrees) * trans(*-center) * trans(j0, i0)
        return new_img[i0:i1, j0:j1], xform

def straighten_image(img, aux=None):
    """Straighten an image returned by planet.
    (Image is assumed to be rotated with blank corners)

    Typical Usage:

        degrees_rotated, straightened_image = straighten_image(image)

    """
    blank = (img[:, :, 3] == 0)
    deg = find_angle(blank)
    if deg > 45:
        deg = deg - 90
    if aux is not None:
        return deg, rotate_and_clip(img, blank, deg, aux=aux)
    else:
        return rotate_and_clip(img, blank, deg) + (deg,) 


def push_scene_info_to_bq(scene_ids_and_paths, table, trys=10, delay_s=10.0, overwrite=False):
    from . import img_utils as iutils
    if len(scene_ids_and_paths) == 0:
        return
    scene_info = None
    for sid, base_path in scene_ids_and_paths:
        [xml_path] = glob(os.path.join(base_path, 'raw', sid + '*_metadata.xml'))
        for _ in range(trys):
            info = scene_info_from_path(xml_path)
            if info is not None:
                break
            time.sleep(delay_s)
        else:
            continue
        if scene_info is None:
            scene_info = {k : [] for k in info}
            scene_info['scene_id'] = []
        for k, v in info.items():
            if k == 'boundary':
                v = json.dumps(v)
            scene_info[k].append(v)
        scene_info['scene_id'].append(sid)
    if scene_info is None:
        return
    scene_info_df = pd.DataFrame(scene_info, columns=sorted(scene_info))
    if_exists = 'replace' if overwrite else 'append'    
    scene_info_df.to_gbq(table, proj_id, if_exists=if_exists)


ns = {
    'gis'  : 'http://www.opengis.net/gml',
    'plnt' : 'http://schemas.planet.com/ps/v1/planet_product_metadata_geocorrected_level',
    'gml'  : 'http://www.opengis.net/gml',
    'eop'  : 'http://earth.esa.int/eop',
    'ps'   : 'http://schemas.planet.com/ps/v1/planet_product_metadata_geocorrected_level',
}

    
def scene_info_from_path(xml_path):
    xml = ET.parse(xml_path).getroot()
    epsg = xml.find('gml:resultOf', ns).find('ps:EarthObservationResult', ns).find(
                'eop:product', ns).find('ps:ProductInformation', ns).find(
               'ps:spatialReferenceSystem', ns).find('ps:epsgCode', ns).text

    t0 = xml.find('gis:validTime', ns).find('gis:TimePeriod', ns).find('gis:beginPosition', ns).text
    t1 = xml.find('gis:validTime', ns).find('gis:TimePeriod', ns).find('gis:endPosition', ns).text
    assert t0 == t1, (t0, t1)

    boundary_str = (xml.find('gis:target', ns)
                       .find('plnt:Footprint', ns)
                       .find('gml:multiExtentOf', ns)
                       .find('gml:MultiSurface', ns)
                       .find('gml:surfaceMembers', ns)
                       .find('gml:Polygon', ns)
                       .find('gml:outerBoundaryIs', ns)
                       .find('gml:LinearRing', ns)
                       .find('gml:coordinates', ns).text
                   )
    pairs = boundary_str.strip().split()
    boundary = []
    for p in pairs:
        lon, lat = p.split(',')
        boundary.append({'lat' : float(lat), 'lon': float(lon)})

    return {
        'lat_min': min([x['lat'] for x in boundary]),
        'lat_max': max([x['lat'] for x in boundary]),
        'lon_min': min([x['lon'] for x in boundary]),
        'lon_max': max([x['lon'] for x in boundary]),
        'timestamp': t0,
        'epsg': epsg,
        'boundary' : boundary,
    }




