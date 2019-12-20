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
import tensorflow.contrib.tensorrt as trt
import tensorflow.compat.v1 as tf
import numpy as np
import json


class ObjectTrackingConfig:

  def __init__(self, score_threshold=0.0, max_detections=100):
    self.score_threshold = score_threshold
    self.max_detections = max_detections


class ObjectTrackingInference:

  def __init__(self, frozen_graph_path, label_map_path, config):
    self.config = config

    self._loadLabelMap(label_map_path)
    self._loadFrozenGraph(frozen_graph_path)

  def __del__(self):
    self.session.close()

  @staticmethod
  def TFTRTModel(*args, **kwargs):
    return ObjectTrackingInference(*args, **kwargs)

  def _loadLabelMap(self, label_map_path):
    with tf.gfile.GFile(label_map_path, 'r') as f:
      # Converts the pbtxt into a JSON file and then parses it.
      parsed_label_map = json.loads(
          '[' + f.read().replace('\n}\nitem {\n', '\n}, \n{\n').replace(
              "\n  name: \"", "\n  \"name\": \"").replace(
                  '\n  id: ', ",\n  \"id\": ")[5:] + ']')

    self.label_map = dict()
    for item in parsed_label_map:
      self.label_map[item['id']] = item['name']

  def _loadFrozenGraph(self, frozen_graph_path):
    trt_graph = tf.GraphDef()
    with tf.gfile.GFile(frozen_graph_path, 'rb') as f:
      trt_graph.ParseFromString(f.read())

    self.graph = tf.Graph()
    with self.graph.as_default():
      self.output_node = tf.import_graph_def(
          trt_graph,
          return_elements=[
              'detection_boxes:0', 'detection_classes:0', 'detection_scores:0',
              'num_detections:0', 'raw_outputs/lstm_c:0', 'raw_outputs/lstm_h:0'
          ])
    self.session = tf.InteractiveSession(graph=self.graph)

    tf_scores = self.graph.get_tensor_by_name('import/detection_scores:0')
    tf_boxes = self.graph.get_tensor_by_name('import/detection_boxes:0')
    tf_classes = self.graph.get_tensor_by_name('import/detection_classes:0')
    tf_num_detections = self.graph.get_tensor_by_name('import/num_detections:0')
    tf_lstm_c = self.graph.get_tensor_by_name('import/raw_outputs/lstm_c:0')
    tf_lstm_h = self.graph.get_tensor_by_name('import/raw_outputs/lstm_h:0')

    self._output_nodes = [
        tf_scores, tf_boxes, tf_classes, tf_num_detections, tf_lstm_c, tf_lstm_h
    ]

    self.lstm_c = np.ones((1, 8, 8, 320))
    self.lstm_h = np.ones((1, 8, 8, 320))

  def getInputSize(self):
    return Size(256, 256)

  def run(self, timestamp, frame, annotations):
    with self.graph.as_default():
      (detection_scores, detection_boxes, detection_classes, num_detections,
       self.lstm_c, self.lstm_h) = self.session.run(
           self._output_nodes,
           feed_dict={
               'import/image_tensor:0': np.array(frame)[None, ...],
               'import/raw_inputs/init_lstm_c:0': self.lstm_c,
               'import/raw_inputs/init_lstm_h:0': self.lstm_h
           })

    boxes = detection_boxes[0]  # index by 0 to remove batch dimension
    scores = detection_scores[0]
    classes = detection_classes[0]
    num_detections = num_detections

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

  def getTFSession(self):
    return self.session


class Size:

  def __init__(self, width, height):
    self.width = width
    self.height = height


class NormalizedBoundingBox:

  def __init__(self, left, top, right, bottom):
    self.left = left
    self.top = top
    self.right = right
    self.bottom = bottom


class ObjectTrackingAnnotation:

  def __init__(self, timestamp, track_id, class_id, class_name,
               confidence_score, bbox):
    self.timestamp = timestamp
    self.track_id = track_id
    self.class_id = class_id
    self.class_name = class_name
    self.confidence_score = confidence_score
    self.bbox = bbox
