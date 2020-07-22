# Copyright 2017 Google Inc. and Global Fishing Watch Inc.
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

import logging
import setuptools
import subprocess
import glob
import os
import distutils
from setuptools.command.install import install
from distutils.command.build import build as _build

DEPENDENCIES = [
    "ujson",
    "NewlineJSON",
    "scikit-image",
    "scikit-learn>=0.18",
    "planet",
    "matplotlib<3",
    "affine",
    "pandas",
    "pandas-gbq",
    "google-cloud-storage",
    'matplotlib_scalebar',
    'tqdm',
    'google-compute-engine',
    'imagecodecs',
    'six'
    ]

print(__file__, __name__)

# This class handles the pip install mechanism.
class build(_build):  # pylint: disable=invalid-name
  """A build command class that will be invoked during package install.
  The package built using the current setup.py will be staged and later
  installed in the worker using `pip install package'. This class will be
  instantiated during install for this specific scenario and will trigger
  running the custom commands specified.
  """
  sub_commands = _build.sub_commands + [('CustomCommands', None)]

# The output of custom commands (including failures) will be logged in the
# worker-startup log.
CUSTOM_COMMANDS = [
    ['apt-get', 'update'],
    ['apt-get', '--assume-yes', 'install', 'python-gdal'],
    ['apt-get', 'install', 'fonts-freefont-ttf'],
]


class CustomCommands(setuptools.Command):
  """A setuptools Command class able to run arbitrary commands."""

  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def RunCustomCommand(self, command_list):
    print('Running command: %s' % command_list)
    p = subprocess.Popen(
        command_list,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Can use communicate(input='y\n'.encode()) if the command run requires
    # some confirmation.
    stdout_data, _ = p.communicate()
    print('Command output: %s' % stdout_data)
    if p.returncode != 0:
      raise RuntimeError(
          'Command %s failed: exit code: %s' % (command_list, p.returncode))

  def run(self):
    for command in CUSTOM_COMMANDS:
      self.RunCustomCommand(command)



setuptools.setup(
    name='optical_vessel_detection',
    version='0.10',
    author='Tim Hochberg',
    author_email='tim@globalfishingwatch.org',
    packages=[
        'optical_vessel_detection',
        'optical_vessel_detection.core',
        'optical_vessel_detection.support',
    ],
    install_requires=DEPENDENCIES ,
    cmdclass={
        # Command class instantiated and run during pip install scenarios.
        'build': build,
        'CustomCommands': CustomCommands,
    },
    extras_require={
        'tf': ['tensorflow>=1.11.0'],
        'tf_gpu': ['tensorflow-gpu>=1.11.0'],
    },    
)