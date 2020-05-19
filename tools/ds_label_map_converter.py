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
r"""Converts an AutoML label map to a DeepStream compatible format.

python3 tools/ds_label_map_converter.py \
  --label_map data/traffic_label_map.pbtxt \
  --output data/output_label_map.txt \
"""

import argparse
from automl_video_ondevice import utils


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--label_map', help='.pbtxt AutoML generated label map', required=True)
  parser.add_argument(
      '--output', help='output DeepStream compatible label map', required=True)
  args = parser.parse_args()

  with open(args.label_map, 'r') as f:
    label_map_dict, label_map_list = utils.parse_label_map(f.read())
    # If there is no label for index 0, fill with 'unknown.'
    # VOT skips index 0 labels, but VCN has index 0.
    if 0 not in label_map_dict:
      label_map_list = ['unknown'] + label_map_list
    label_map_list = label_map_list + ['']
    label_map_txt = '\n'.join(label_map_list)

  with open(args.output, 'w') as f:
    f.write(label_map_txt)


if __name__ == '__main__':
  main()
