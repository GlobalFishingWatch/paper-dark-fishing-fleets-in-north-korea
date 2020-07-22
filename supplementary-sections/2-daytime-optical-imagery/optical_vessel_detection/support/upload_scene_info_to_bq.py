from __future__ import print_function
from __future__ import division
import datetime
import dateutil.parser
from glob import glob
import os
import sys

from .core import planet_utils as putils


this_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(this_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", action="append", dest="dates",
                        help="date to download images for")
    parser.add_argument("--overwrite", action="store_true", dest="overwrite",
                        help="if specified, overwrite target table")
    parser.add_argument("--table", default="machine_learning_dev_ttl_120d.planet_scene_info")
    args = parser.parse_args()
    base_path = os.path.join(parent_dir, 'nk_detection')

    if args.dates:
        timestamps = []
        for x in args.dates:
            ts = dateutil.parser.parse(x)
            assert ts.time() == datetime.time()
            timestamps.append(ts)
    else:
        paths = glob(os.path.join(base_path, '*'))
        names = [os.path.basename(x) for x in paths]
        timestamps = [dateutil.parser.parse(x) for x in names]

    scene_ids_and_paths = []
    for ts in timestamps:
        sys.stdout.flush()
        dir_name = "{:%Y%m%d}".format(ts)
        date_path = os.path.join(base_path, dir_name)
        vessel_paths = glob(os.path.join(date_path, 'raw', '*.xml'))
        for x in vessel_paths:
            scene_id = os.path.basename(x).rsplit('_', 2)[0]
            scene_ids_and_paths.append((scene_id, date_path))
    putils.push_scene_info_to_bq(scene_ids_and_paths, args.table, overwrite=args.overwrite)
