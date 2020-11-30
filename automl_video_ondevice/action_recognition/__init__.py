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
"""Action recognition loader.

The action recognition model can deviate further from classification as time
goes on. For now, inference is identical to shot classification so this serves
as an alias.

Based on filename, the loader will instantiate an inference engine.
"""

from automl_video_ondevice import shot_classification
from automl_video_ondevice.shot_classification.config import ShotClassificationConfig as ActionRecognitionConfig
from automl_video_ondevice.types import Format
from automl_video_ondevice.types import ShotClassificationAnnotation
from automl_video_ondevice.types import Size
from automl_video_ondevice.utils import format_from_filename

load = shot_classification.load
