#!/usr/bin/env python

# Copyright 2017 Google Inc. and Skytruth Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import yaml
import json
import time
import subprocess
import os
import datetime
import argparse
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
import tempfile


def launch(model_name, job_name, config_file, extra_args):
    # Read the configuration file so that we 
    # know the train path and don't need to
    # hardcode it here
    with open(config_file) as f:
        config = yaml.load(f.read())
        tf_config_template = config['tensor_flow_config_template']

    extra_args_str = ',\n'.join(['"{}"'.format(x) for x in extra_args])

    tf_config_txt = tf_config_template.format(job_name=job_name, extra_args=extra_args_str)

    start_time = datetime.datetime.utcnow()
    timestamp = start_time.strftime('%Y%m%dT%H%M%S')
    job_id = ('%s_%s_%s' % (model_name, job_name, timestamp)).replace(
        '.', '_').replace('-', '_')

    # Kick off the job on CloudML
    with tempfile.NamedTemporaryFile() as temp:
        temp.write(tf_config_txt)
        temp.flush()

        with open(temp.name) as f:
            tf_config = yaml.load(f)

        # It seems that we currently need to pass args as both 'args' in the
        # config file and as args after the '--'?!
        args = [
            'gcloud', 'ml-engine', 
            'jobs', 'submit', 'training', job_id,
            '--config', temp.name, '--module-name',
            "optical_vessel_detection.{}".format(model_name), '--staging-bucket',
            config['staging_bucket'], '--package-path', 'optical_vessel_detection',
            '--region', config['region'], '--'
        ] + tf_config['trainingInput']['args']

        print('Executing:\n', ' '.join(args))
        print("Config:\n", tf_config_txt)

        subprocess.check_call(args)

    return job_id


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Deploy ML Training.')
    parser.add_argument('--model_name', required=True, 
                        help='module name of model.')
    parser.add_argument('--job_name', required=True, 
                        help='unique name for this job.')
    parser.add_argument("--initial-step", type=int, 
                        help="step (for weight decay, etc) to start training from")
    parser.add_argument("--warm-start-from", 
                        help="path to weights to warm start from")
    parser.add_argument("--add-date", 
                        help="use all data for training (validation data will be invalid)")
    parser.add_argument( '--config_file', required=True,
                        help='configuration file path.')
    parser.add_argument("--split-type", default="by-scene",
                        help="'by-scene'|'by-tile';"
                             "by tile is optimistic but has more coverage")
    parser.add_argument("--split", default=4, type=int,
                        help="0-4, or -1 to train on all data, but use split 0 as (bogus) validation")
    args = parser.parse_args()

    extra_args = []
    if args.initial_step:
        extra_args.extend(['--initial-step', args.initial_step])
    if args.warm_start_from:
        extra_args.extend(['--warm-start-from', args.warm_start_from])
    if args.split_type:
        extra_args.extend(['--split-type', args.split_type])
    if args.split:
        extra_args.extend(['--split', args.split])



    launch(args.model_name, args.job_name, args.config_file, extra_args)
