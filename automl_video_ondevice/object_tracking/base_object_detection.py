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
"""Provides the base class for implementing video object tracking inference."""

from automl_video_ondevice.types import Size


class BaseObjectDetectionInference:
  """Interface that must be implemented for support of different model types."""

  def __init__(self, frozen_graph_path, label_map_path, config):
    raise NotImplementedError()

  def input_size(self):
    return Size(256, 256)

  def run(self, timestamp, frame, annotations):
    raise NotImplementedError()
