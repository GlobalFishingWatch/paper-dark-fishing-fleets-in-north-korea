from dateutil import parser
import datetime
import os
import pandas as pd
from glob import glob
from skimage import io as skio
import numpy as np
import tensorflow as tf
from skimage import morphology
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os
import json
from skimage import draw
from skimage import morphology
import logging

this_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(this_dir)
grandparent_dir = os.path.dirname(parent_dir)

proj_id = os.environ['PROJ_ID']

ais_info_template = '''
#standardSQL

SELECT
  *
FROM (
  SELECT
    (lat*timeto2 + lat2*timeto)/(timeto+timeto2) lat_center,
    (lon*timeto2 + lon2*timeto)/(timeto+timeto2) lon_center,
    *
  FROM (
    SELECT
      mmsi,
      timestamp_diff(timestamp2,
        {start_time},
        SECOND) timeto2,
      timestamp_diff({start_time},
        timestamp,
        SECOND) timeto,
      timestamp,
      timestamp2,
      {start_time} start_time,
      lat,
      lat2,
      lon,
      lon2,
      speed,
      speed2,
      course,
      course2
    FROM (
      SELECT
        ssvid as mmsi,
        lat,
        LEAD(lat,1) OVER (PARTITION BY seg_id ORDER BY timestamp) lat2,
        lon,
        LEAD(lon,1) OVER (PARTITION BY seg_id ORDER BY timestamp) lon2,
        timestamp,
        LEAD(timestamp,1) OVER (PARTITION BY seg_id ORDER BY timestamp) timestamp2,
        speed,
        LEAD(speed,1) OVER (PARTITION BY seg_id ORDER BY timestamp) speed2,
        course,
        LEAD(course,1) OVER (PARTITION BY seg_id ORDER BY timestamp) course2
      FROM
        `{ais_table}*`
      WHERE
        _TABLE_SUFFIX BETWEEN '{min_date:%Y%m%d}' AND '{max_date:%Y%m%d}'
    )
    WHERE
          lat >= {lat_min}
      AND lat <= {lat_max}
      AND lon >= {lon_min}
      AND lon <= {lon_max}
      AND timestamp <= {start_time}
      AND timestamp2 >= {start_time}
      AND (timestamp  >= TIMESTAMP('{min_valid_time}')
        OR timestamp2 <= TIMESTAMP('{max_valid_time}'))

       ))
'''


def load_ais_info(metadata, ais_table):
    """Load AIS pings for a scene that occur near the same time as the image was takes

    Parameters
    ----------
    metadata : metadata as returned by iutils.scene_info_from_path

    Returns
    -------
    pd.DateFrame
        Dataframe containing AIS message info for the nearest ping before and after the 
        the image was taken.

    """
    timestamp = parser.parse(metadata['timestamp'])
    pre_time = timestamp - datetime.timedelta(minutes=30)
    post_time = timestamp + datetime.timedelta(minutes=30)
    min_date = timestamp - datetime.timedelta(days=1)
    max_date = timestamp + datetime.timedelta(days=1)
    q = ais_info_template.format(
                    start_time = "TIMESTAMP('{timestamp}')".format(**metadata),
                    YYYYMM = metadata['timestamp'].replace('-', '')[:6],
                    min_valid_time=pre_time,
                    max_valid_time=post_time,
                    min_date = min_date,
                    max_date = max_date,
                    ais_table = ais_table,
                    **metadata)
    return pd.read_gbq(q, project_id=proj_id, dialect='standard')


def overlay_ais_on_sceneinfo(scene_info, ais_df, fontsize=36, dilations=1):
    """Load AIS pings for a scene that occur near the same time as the image was takes

    Parameters
    ----------
    scene_info : SceneInfo
    ais_df : pd.DataFrame
        As returned by `load_ais_info(scene_info)

    Returns
    -------
    SceneInfo
        Copy of `scene_info` with AIS locations overlaid on the image
    list of dicts
        Json encoded information about the overlaid AIS location. Item format:
            {
            'lon' : lon, 
            'lat' : lat, 
            'extrapolated_lon' : lon1,
            'extrapolated_lat' : lat1,
            'dt_s': dt,
            'mmsi': mmsi
            }
    """
    scene = scene_info.scene
    scene_shape = scene.shape
    scene_info = scene_info._replace(scene=None) # Save some memory
    locs = []
    for x in ais_df.itertuples():
        if x.timeto < x.timeto2:
            lat, lon, course, speed, dt = x.lat, x.lon, x.course, x.speed, -x.timeto
        else:
            lat, lon, course, speed, dt = x.lat2, x.lon2, x.course2, x.speed2, x.timeto2
        mmsi = x.mmsi
        angle = np.radians(course) # Relative to north pole, clockwise.
        dh = dt / (60. * 60.) # delta hours
        dl = -speed * dh # distance nautical miles
        dy = np.cos(angle) * dl
        dx = np.sin(angle) * dl
        dlat = dy / 60.0 # 1 nm = 1 minute of latitude
        dlon = dx / 60.0 / np.cos(np.radians(lat)) # 1 nm = 1 minute of long at equator, scaled by cos(lat)
        locs.append((lon, lat, dt, lon + dlon, lat + dlat, str(mmsi)))

    valid_locs = []
    for l in locs:
        (lon, lat, _, lon1, lat1, _) = l
        c, r = [int(round(x)) for x in scene_info.lonlat_to_cr(lon1, lat1)]
        if 0 <= c < scene_shape[1] and 0 <= r <= scene_shape[0]:
            valid_locs.append(l)

    for (lon0, lat0, _, lon1, lat1, _) in valid_locs:
        c0, r0 = [int(round(x)) for x in 
                    scene_info.lonlat_to_cr(lon0, lat0)]
        rr, cc = draw.circle(r0, c0, 10, shape=scene_shape)
        scene[rr, cc] = (1, 1, 1, 1)
        c, r = [int(round(x)) for x in 
                    scene_info.lonlat_to_cr(lon1, lat1)]
        rr, cc = draw.circle(r, c, 10, shape=scene_shape)
        scene[rr, cc] = (1, 1, 1, 1)
        rr1, cc1 = draw.line(r0, c0, r, c)
        rr1a = [r for (r, c) in zip(rr1, cc1)
                    if 0 <= r < scene_shape[0] and
                       0 <= c < scene_shape[1]]
        cc1a = [c for (r, c) in zip(rr1, cc1)
                    if 0 <= r < scene_shape[0] and
                       0 <= c < scene_shape[1]]
        mask = np.zeros(scene.shape[:2], dtype=bool)
        mask[rr1a, cc1a] = 1
        for i in range(dilations):
            mask = morphology.binary_dilation(mask)
        scene[mask] = (1, 1, 1, 1)
        rr, cc = draw.circle(r, c, 7, shape=scene_shape)
        scene[rr, cc] = (0, 0, 0, 1)

    img = Image.fromarray(np.round(scene * 255).astype(np.uint8))
    del scene # Save some memory
    drawer = ImageDraw.Draw(img)
    # Need sudo apt-get install fonts-freefont-ttf
    try:
        fnt = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMono.ttf', size=fontsize)
    except:
        fnt = ImageFont.truetype(os.path.join(grandparent_dir, 'FreeMono.ttf'), size=fontsize)

    json_locs = []
    for (lon, lat, dt, lon1, lat1, mmsi) in valid_locs:

        c, r = [int(round(x)) for x in scene_info.lonlat_to_cr(lon, lat)]
        c1, r1 = [int(round(x)) for x in scene_info.lonlat_to_cr(lon1, lat1)] 
        text = mmsi + "({:+.0f}min)".format(dt/60.0) 
        if c1 > c:
            sz = drawer.textsize(text, fnt)
            c -= (sz[0] + 10)
        else:
            c += 10
        c = np.clip(c, 20, scene_shape[1] - 50)
        r = np.clip(r, 20, scene_shape[0] - 50)
        color = (255, 255, 255)
        drawer.text((c, r - 15), text, color, font=fnt)
        json_locs.append({
            'lon' : lon, 'lat' : lat, 
            'extrapolated_lon' : lon1,
            'extrapolated_lat' : lat1,
            'dt_s': dt,
            'mmsi': mmsi
            })

    scene_info = scene_info._replace(scene=np.array(img))
    return scene_info, json_locs









