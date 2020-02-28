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
"""Shared utility functions for each demo."""

import time
import cv2
import numpy as np

colors = [(np.random.uniform(0, 255), np.random.uniform(0, 255),
           np.random.uniform(0, 255)),
          (np.random.uniform(0, 255), np.random.uniform(0, 255),
           np.random.uniform(0, 255)),
          (np.random.uniform(0, 255), np.random.uniform(0, 255),
           np.random.uniform(0, 255)),
          (np.random.uniform(0, 255), np.random.uniform(0, 255),
           np.random.uniform(0, 255)),
          (np.random.uniform(0, 255), np.random.uniform(0, 255),
           np.random.uniform(0, 255)),
          (np.random.uniform(0, 255), np.random.uniform(0, 255),
           np.random.uniform(0, 255))]


def render_bbox(image, annotations):
  """Renders a visualzation of bounding box annotations.

  Args:
    image: numpy array of image bytes.
    annotations: the annotations to render onto the image.

  Returns:
    numpy array of image, with bounding box drawn.
  """
  height, width, _ = image.shape
  for annotation in annotations:
    x0, y0, x1, y1 = (annotation.bbox.left, annotation.bbox.top,
                      annotation.bbox.right, annotation.bbox.bottom)
    # Converts coordinates from relative [0,1] space to [width,height]
    x0, y0, x1, y1 = int(x0 * width), int(y0 * height), int(x1 * width), int(
        y1 * height)
    percent = int(100 * annotation.confidence_score)
    label = '%d%% %s' % (percent, annotation.class_name)

    if annotation.track_id == -1:
      color = (0, 255, 0)
    else:
      color = colors[annotation.track_id % len(colors)]

    image = cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    image = cv2.putText(image, label, (x0, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 0, 0), 2)
  return image


class FpsCalculator:
  """Calculates FPS and Latency of the current loop."""

  def __init__(self):
    self.fps_time_queue = []

  def measure(self):
    """Conducts one step of time measurement.

    Returns:
      fps: The frames per second.
      latency: How long it's been since the last measure() call, in ms.
    """
    timestamp = time.time()
    self.fps_time_queue.append(timestamp)
    if len(self.fps_time_queue) > 15:
      self.fps_time_queue.pop(0)
    if len(self.fps_time_queue) > 2:
      fps = round(
          len(self.fps_time_queue) /
          (self.fps_time_queue[-1] - self.fps_time_queue[0]), 2)
      latency = round(self.fps_time_queue[-1] - self.fps_time_queue[-2],
                      4) * 1000
    else:
      fps = 'N/A'
      latency = 'N/A'
    return fps, latency
