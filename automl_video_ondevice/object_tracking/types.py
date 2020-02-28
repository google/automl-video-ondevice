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
"""The structs and types used for object tracking."""

import enum
import dataclasses


@dataclasses.dataclass
class Size:
  width: int
  height: int


@dataclasses.dataclass
class NormalizedBoundingBox:
  left: float
  top: float
  right: float
  bottom: float


@dataclasses.dataclass
class ObjectTrackingAnnotation:
  timestamp: float
  track_id: int
  class_id: int
  class_name: str
  confidence_score: float
  bbox: NormalizedBoundingBox


class Format(enum.Enum):
  UNDEFINED = 0
  TFLITE = 1
  TENSORFLOW = 2


class Tracker(enum.Enum):
  NONE = 0
  FAST_INACCURATE = 1
  BASIC = 2
  HIGH_QUALITY_SLOW = 3
