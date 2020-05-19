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
"""Provides an implementation of object tracking using EdgeTPU TFLite."""

from os import path
import numpy as np
try:
  import tflite_runtime.interpreter as tflite
except:
	try:
		import tensorflow.lite as tflite
	except:
		print("Can't find the TFLite runtime. Follow directions here: https://www.tensorflow.org/lite/guide/python")
import platform
from automl_video_ondevice.object_tracking.base_object_detection import BaseObjectDetectionInference
from automl_video_ondevice.types import NormalizedBoundingBox
from automl_video_ondevice.types import ObjectTrackingAnnotation
from automl_video_ondevice.types import Size

import automl_video_ondevice.utils as vot_utils

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


class TFLiteObjectDetectionInference(BaseObjectDetectionInference):
  """Implementation of the BaseObjectDetectionInference using EdgeTPU / TFLite.

  EdgeTPU TFLite is a hybrid solution, converting a sub-section of the TFLite
  graph into an EdgeTPU-optimized operator.

  Regular TFLite can also be used with this implementation.
  """

  def __init__(self, tflite_path, label_map_path, config):
    self._config = config

    self._load_label_map(label_map_path)
    self._load_tflite(tflite_path)

  def _load_label_map(self, label_map_path):
    with open(label_map_path, 'r') as f:
      _, self.label_list = vot_utils.parse_label_map(f.read())

  def _load_tflite(self, tflite_path):
    experimental_delegates = []
    try:
      experimental_delegates.append(
          tflite.load_delegate(
              EDGETPU_SHARED_LIB,
              {'device': self._config.device} if self._config.device else {}))
    except AttributeError as e:
      if '\'Delegate\' object has no attribute \'_library\'' in str(e):
        print(
            'Warning: EdgeTPU library not found. You can still run CPU models, '
            'but if you have a Coral device make sure you set it up: '
            'https://coral.ai/docs/setup/.')
    except ValueError as e:
      if 'Failed to load delegate from ' in str(e):
        print(
            'Warning: EdgeTPU library not found. You can still run CPU models, '
            'but if you have a Coral device make sure you set it up: '
            'https://coral.ai/docs/setup/.')

    try:
      self._interpreter = tflite.Interpreter(
          model_path=tflite_path, experimental_delegates=experimental_delegates)
    except TypeError as e:
      if 'got an unexpected keyword argument \'experimental_delegates\'' in str(
          e):
        self._interpreter = tflite.Interpreter(model_path=tflite_path)
    try:
      self._interpreter.allocate_tensors()
    except RuntimeError as e:
      if 'edgetpu-custom-op' in str(e) or 'EdgeTpuDelegateForCustomOp' in str(
          e):
        raise RuntimeError('Loaded an EdgeTPU model without the EdgeTPU '
                           'library loaded. If you have a Coral device make '
                           'sure you set it up: https://coral.ai/docs/setup/.')
      else:
        raise e
    self._is_lstm = self._check_lstm()
    if self._is_lstm:
      print('Loading an LSTM model.')
      self._lstm_c = np.copy(self.input_tensor(1))
      self._lstm_h = np.copy(self.input_tensor(2))

  def _check_lstm(self):
    return len(self._interpreter.get_input_details()) > 1 and len(
        self._interpreter.get_output_details()) > 4

  def input_size(self):
    _, height, width, _ = self._interpreter.get_input_details()[0]['shape']
    return Size(width, height)

  def input_tensor(self, index):
    tensor_index = self._interpreter.get_input_details()[index]['index']
    return self._interpreter.tensor(tensor_index)()[0]

  def output_tensor(self, index):
    tensor_index = self._interpreter.get_output_details()[index]['index']
    tensor = self._interpreter.tensor(tensor_index)()
    return np.squeeze(tensor)

  def fill_inputs(self, frame):
    input_image = self.input_tensor(0)
    if self._is_lstm:
      input_lstm_c = self.input_tensor(1)
      input_lstm_h = self.input_tensor(2)

    np.copyto(input_image, frame)
    if self._is_lstm:
      np.copyto(input_lstm_c, self._lstm_c)
      np.copyto(input_lstm_h, self._lstm_h)

  def run(self, timestamp, frame, annotations):
    # Interpreter hates it when native tensors are retained.
    # fill_inputs will release input tensors after filling with data.
    self.fill_inputs(frame)
    self._interpreter.invoke()

    boxes = self.output_tensor(0)
    classes = self.output_tensor(1)
    scores = self.output_tensor(2)
    num_detections = self.output_tensor(3)
    if self._is_lstm:
      output_lstm_c = self.output_tensor(4)
      output_lstm_h = self.output_tensor(5)

      np.copyto(self._lstm_c, output_lstm_c)
      np.copyto(self._lstm_h, output_lstm_h)

    for i in range(int(num_detections)):
      box = boxes[i]

      if scores[i] > self._config.score_threshold:

        bbox = NormalizedBoundingBox(
            left=box[1], top=box[0], right=box[3], bottom=box[2])

        annotation = ObjectTrackingAnnotation(
            timestamp=timestamp,
            track_id=-1,
            class_id=int(classes[i]),
            class_name=self.label_list[int(classes[i])],
            confidence_score=scores[i],
            bbox=bbox)

        annotations.append(annotation)

    return True
