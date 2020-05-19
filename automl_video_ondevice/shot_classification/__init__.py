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
"""Object tracking loader.

Based on filename, the loader will instantiate an inference engine.
"""

from automl_video_ondevice.shot_classification.base_shot_classification import BaseShotClassificationInference
from automl_video_ondevice.shot_classification.config import ShotClassificationConfig
from automl_video_ondevice.types import Format
from automl_video_ondevice.utils import format_from_filename


def load(frozen_graph_path,
         label_map_path,
         config,
         file_format=Format.UNDEFINED):
  
  # type: (str, str, ShotClassificationConfig, Format) -> BaseObjectDetectionInference
  
  """Instantiates an inference engine based on the file format.

  Args:
    frozen_graph_path: Path to the model frozen graph to be used.
    label_map_path: Path to the labelmap .pbtxt file.
    config: An ObjectTrackingConfig instance.
    file_format: Specifies which format the graph is in. If undefined, will make
      assumptions based on filename.

  Returns:
    An instantiated inference engine.
  """
  del config  # Unused.

  if file_format == Format.UNDEFINED:
    file_format = format_from_filename(frozen_graph_path)

  print('Loading: {} <{}> {}'.format(frozen_graph_path, file_format,
                                     label_map_path))

  return BaseShotClassificationInference(None, None, None)
