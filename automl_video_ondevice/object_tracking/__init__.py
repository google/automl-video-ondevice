# Lint as: python3
# Copyright 2019 Google LLC
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

from automl_video_ondevice.object_tracking.base_object_detection import BaseObjectDetectionInference
from automl_video_ondevice.object_tracking.camshift_object_tracker import CamshiftObjectTracker
from automl_video_ondevice.object_tracking.config import ObjectTrackingConfig
from automl_video_ondevice.object_tracking.types import Format
from automl_video_ondevice.object_tracking.types import NormalizedBoundingBox
from automl_video_ondevice.object_tracking.types import ObjectTrackingAnnotation
from automl_video_ondevice.object_tracking.types import Size
from automl_video_ondevice.object_tracking.types import Tracker
from automl_video_ondevice.object_tracking.utils import format_from_filename


def load(frozen_graph_path,
         label_map_path,
         config,
         file_format=Format.UNDEFINED):
  
  # type: (str, str, ObjectTrackingConfig, Format) -> BaseObjectDetectionInference
  
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
  if file_format == Format.UNDEFINED:
    file_format = format_from_filename(frozen_graph_path)

  print('Loading: {} <{}> {}'.format(frozen_graph_path, file_format,
                                     label_map_path))

  engine = None

  # Some modules may never even be loaded. Only hotloads what is necessary.
  
  if file_format == Format.TFLITE:
    from automl_video_ondevice.object_tracking.tflite_object_detection import TFLiteObjectDetectionInference
    engine = TFLiteObjectDetectionInference(frozen_graph_path, label_map_path,
                                            config)
  elif file_format == Format.TENSORFLOW:
    from automl_video_ondevice.object_tracking.tf_object_detection import TFObjectDetectionInference
    engine = TFObjectDetectionInference(frozen_graph_path, label_map_path,
                                        config)
  else:
    engine = BaseObjectDetectionInference(None, None, None)
  

  if config.tracker == Tracker.FAST_INACCURATE:
    return CamshiftObjectTracker(engine, config)
  elif not config.tracker or config.tracker == Tracker.NONE:
    return engine
  else:
    raise NotImplementedError('Invalid or unimplemented tracker type.')
