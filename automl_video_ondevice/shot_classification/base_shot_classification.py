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

from typing import List
from typing import Union
import numpy as np

from automl_video_ondevice.shot_classification.config import ShotClassificationConfig
from automl_video_ondevice.types import ShotClassificationAnnotation
from automl_video_ondevice.types import Size


class BaseShotClassificationInference:
  """Interface that must be implemented for support of different model types."""

  def __init__(self, frozen_graph_path: str, label_map_path: str,
               config: ShotClassificationConfig):
    """Constructor for BaseShotClassificationInference.

    Args:
      frozen_graph_path: String value for the file path of frozen graph.
      label_map_path: String value for the file path of the label map.
      config: ShotClassificationConfig object with shot classification configs.
    """
    raise NotImplementedError()

  def input_size(self) -> Size:
    """Calculate / grab optimal input size.

    The user is expected to ensure the size of their input image is correct.
    This is in case the user wants to do any acceleration of image resizing
    themselves.

    Some inference engines require a specific input image size such as the
    TFLite models, however some Tensorflow models accept a dynamic input. For
    a dynamic input, the size outputed will have the dimensions -1, -1.

    Returns:
      The expected input size, of the type Size.
    """
    return Size(256, 256)

  def run(self, timestamp: Union[int, float], frame: np.ndarray,
          annotations: List[ShotClassificationAnnotation]) -> bool:
    """Run inferencing for a single frame, to calculate annotations.

    Args:
      timestamp: Generally an integer representing the microsecond of the frame,
        however any unique number is also accepted.
      frame: A numpy array of the shape (h, w, 3), representing an RGB image.
        Each color channel should be a number [0, 256).
      annotations: A list to append the output annotations to. For normal use-
        case, this should be an empty list. The output annotations will be of
        type ShotClassificationAnnotation.

    Returns:
      A boolean, True if successful and False if unsuccessful.
    """
    raise NotImplementedError('Shot classification has not been implemented.')
