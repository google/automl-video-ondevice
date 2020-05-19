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
"""Public configuration parameters for object tracking."""

import dataclasses


@dataclasses.dataclass(eq=True)
class ShotClassificationConfig:
  device: str = ""

  # Number of highest scoring labels to output.
  # If max_results is -1 then all labels are used.
  top_k: int = 5

  # Only labels with scores > threshold are output.
  score_threshold: float = 0.0

  # Outputs duplicate results for several frames.
  # Predictions from the classification model is done every 25 frames. If
  # duplicate_results is true then each 25 frame input will return the previous
  # output, otherwise the output array will be have a single None value.
  duplicate_results: bool = False

  # Waits this number of frames before running inference.
  inference_rate: int = 25
