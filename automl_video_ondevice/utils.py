# Lint as: python3
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for object tracking."""

import json
import re

from automl_video_ondevice.types import Format


def parse_label_map(label_map):
  """Provides a short implementation of label map parsing.

  Args:
    label_map: The label map data, as a UTF-8 string.

  Returns:
    The label map dictionary.

    Example output:
      dict {
        1: 'label',
        2: 'label2'
      }
  """
  # Converts the pbtxt into a JSON file and then parses it.
  label_map_json = '[' + re.sub(
      r'[\n\s]*item[\n\s]*', '',
      re.sub(
          r'[\n\s]*}[\n\s]+item[\n\s]*{[\n\s]*', '\n}, \n{\n',
          re.sub(r'[\n\s]*name:\s*"', '\n  "name": "',
                 re.sub(r'\n?\s+id:\s*', ',\n  "id": ', label_map)))) + ']'
  parsed_label_map = json.loads(label_map_json)

  # TensorFlow maps by id to name.
  # TFLite maps by list index to name, ditching the id.

  label_map = dict()
  label_list = list()
  for item in parsed_label_map:
    label_map[item['id']] = item['name']
    label_list.append(item['name'])

  return label_map, label_list


def format_from_filename(filename):
  """Determines format of file from the filename.

  Args:
    filename: The filename to parse.

  Returns:
    The format of the file.
  """
  if filename.endswith('.tflite'):
    return Format.TFLITE
  if filename.endswith('.pb'):
    return Format.TENSORFLOW
  return Format.UNDEFINED
