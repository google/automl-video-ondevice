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
"""Provides an implementation of object tracking using TF and TF-TRT."""

import numpy as np
import tensorflow.compat.v1 as tf
from automl_video_ondevice.object_tracking.base_object_detection import BaseObjectDetectionInference
from automl_video_ondevice.types import NormalizedBoundingBox
from automl_video_ondevice.types import ObjectTrackingAnnotation
from automl_video_ondevice.types import Size

import automl_video_ondevice.utils as vot_utils
import tensorflow.contrib.tensorrt as trt  


class TFObjectDetectionInference(BaseObjectDetectionInference):
  """Implementation of the BaseObjectDetectionInference using TF, or TF-TRT.

  TF-TRT is a hybrid solution, converting a sub-section of the Tensorflow graph
  into it's proprietary TensorRT format. This allows for maximum usage of
  NVIDIA devices while keeping full Tensorflow support.

  Non-TRT frozen graphs can also be used with this implementation.
  """

  def __init__(self, frozen_graph_path, label_map_path, config):
    self.config = config

    self._load_label_map(label_map_path)
    self._load_frozen_graph(frozen_graph_path)

  def _load_label_map(self, label_map_path):
    with open(label_map_path, 'r') as f:
      self.label_map, _ = vot_utils.parse_label_map(f.read())

  def _load_frozen_graph(self, frozen_graph_path):
    trt_graph = tf.GraphDef()
    with open(frozen_graph_path, 'rb') as f:
      trt_graph.ParseFromString(f.read())

    self._is_lstm = self._check_lstm(trt_graph)
    if self._is_lstm:
      print('Loading an LSTM model.')

    self.graph = tf.Graph()
    with self.graph.as_default():
      self.output_node = tf.import_graph_def(
          trt_graph,
          return_elements=[
              'detection_boxes:0', 'detection_classes:0', 'detection_scores:0',
              'num_detections:0'
          ] + (['raw_outputs/lstm_c:0', 'raw_outputs/lstm_h:0']
               if self._is_lstm else []))
    self.session = tf.InteractiveSession(graph=self.graph)

    tf_scores = self.graph.get_tensor_by_name('import/detection_scores:0')
    tf_boxes = self.graph.get_tensor_by_name('import/detection_boxes:0')
    tf_classes = self.graph.get_tensor_by_name('import/detection_classes:0')
    tf_num_detections = self.graph.get_tensor_by_name('import/num_detections:0')
    if self._is_lstm:
      tf_lstm_c = self.graph.get_tensor_by_name('import/raw_outputs/lstm_c:0')
      tf_lstm_h = self.graph.get_tensor_by_name('import/raw_outputs/lstm_h:0')

    self._output_nodes = [tf_scores, tf_boxes, tf_classes, tf_num_detections
                         ] + ([tf_lstm_c, tf_lstm_h] if self._is_lstm else [])

    if self._is_lstm:
      self.lstm_c = np.ones((1, 8, 8, 320))
      self.lstm_h = np.ones((1, 8, 8, 320))

  def _check_lstm(self, graph_def):
    for node in graph_def.node:
      if node.name == 'raw_outputs/lstm_c' or node.name == 'raw_outputs/lstm_h':
        return True
    return False

  def input_size(self):
    return Size(256, 256)

  def run(self, timestamp, frame, annotations):
    with self.graph.as_default():
      # Tensors to feed in.
      feed_dict = {
          'import/image_tensor:0': np.array(frame)[None, ...],
      }
      if self._is_lstm:
        feed_dict.update({
            'import/raw_inputs/init_lstm_c:0': self.lstm_c,
            'import/raw_inputs/init_lstm_h:0': self.lstm_h
        })

      session_return = self.session.run(self._output_nodes, feed_dict=feed_dict)

      # Unpacks tensor output.
      if self._is_lstm:
        (detection_scores, detection_boxes, detection_classes, num_detections,
         self.lstm_c, self.lstm_h) = session_return
      else:
        (detection_scores, detection_boxes, detection_classes,
         num_detections) = session_return

    boxes = detection_boxes[0]  # index by 0 to remove batch dimension
    scores = detection_scores[0]
    classes = detection_classes[0]

    for i in range(int(num_detections)):
      box = boxes[i]

      if scores[i] > self.config.score_threshold:

        bbox = NormalizedBoundingBox(
            left=box[1], top=box[0], right=box[3], bottom=box[2])

        annotation = ObjectTrackingAnnotation(
            timestamp=timestamp,
            track_id=-1,
            class_id=classes[i],
            class_name=self.label_map[classes[i]],
            confidence_score=scores[i],
            bbox=bbox)

        annotations.append(annotation)

    return True
