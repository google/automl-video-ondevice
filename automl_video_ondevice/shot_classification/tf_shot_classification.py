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

import copy
import numpy as np
import tensorflow.compat.v1 as tf
from automl_video_ondevice.shot_classification.base_shot_classification import BaseShotClassificationInference
from automl_video_ondevice.types import ShotClassificationAnnotation

import automl_video_ondevice.utils as vot_utils
import tensorflow.contrib.tensorrt as trt  


class TFShotClassificationInference(BaseShotClassificationInference):
  """Implementation of the BaseShotClassificationInference using TF, or TF-TRT.

  TF-TRT is a hybrid solution, converting a sub-section of the Tensorflow graph
  into it's proprietary TensorRT format. This allows for maximum usage of
  NVIDIA devices while keeping full Tensorflow support.

  Non-TRT frozen graphs can also be used with this implementation.
  """

  def __init__(self, frozen_graph_path, label_map_path, config):
    self.config = config

    if self.config.top_k <= 0:
      raise ValueError(
          'Top k cannot be zero, or else no results will be returned.')

    self._load_label_map(label_map_path)
    self._load_frozen_graph(frozen_graph_path)

  def __del__(self):
    self.session.close()

  def _load_label_map(self, label_map_path):
    with open(label_map_path, 'r') as f:
      self.label_map, _ = vot_utils.parse_label_map(f.read())

  def _load_frozen_graph(self, frozen_graph_path):
    frozen_graph = tf.GraphDef()
    with open(frozen_graph_path, 'rb') as f:
      frozen_graph.ParseFromString(f.read())

    self.graph = tf.Graph()
    with self.graph.as_default():
      self.output_node = tf.import_graph_def(
          frozen_graph, return_elements=[
              'probabilities:0',
          ])
    self.session = tf.InteractiveSession(graph=self.graph)

    tf_probabilities = self.graph.get_tensor_by_name('import/probabilities:0')
    self._output_nodes = [tf_probabilities]
    self.sliding_window = None
    self.frames_since_last_inference = self.config.inference_rate
    self.last_annotations = []

  def run(self, timestamp, frame, annotations):
    np_frame = np.array(frame)

    # User has to handle frame resizing by themselves.
    if np.shape(np_frame)[0] != 256:
      raise ValueError(
          'Shot classification input must be a height of 256 pixels. '
          'There is no width limit. Aspect ratio must be retained.')

    # Initiate sliding window if not created yet.
    if self.sliding_window is None:
      self.sliding_window = np.zeros(
          (64, np.shape(np_frame)[0], np.shape(np_frame)[1], 3))

    # Moves sliding window and adds new frame to the back
    self.sliding_window = np.roll(self.sliding_window, -1)
    self.sliding_window[63] = np_frame

    self.frames_since_last_inference += 1
    if self.frames_since_last_inference >= self.config.inference_rate:
      self.frames_since_last_inference = 0
      with self.graph.as_default():
        (probabilities) = self.session.run(
            self._output_nodes,
            feed_dict={
                'import/video_inputs:0': self.sliding_window,
            })

        score = probabilities[0]

        assert len(self.label_map) == len(score)

        for i in range(len(score)):
          if score[i] < self.config.score_threshold:
            continue

          annotation = ShotClassificationAnnotation(
              timestamp=timestamp,
              class_name=self.label_map[i],
              confidence_score=score[i],
          )
          annotations.append(annotation)
        annotations.sort(key=lambda v: v.confidence_score, reverse=True)

        # Removes everything not in the top k.
        if self.config.top_k > 0:
          del annotations[self.config.top_k:]

        if self.config.duplicate_results:
          self.last_annotations = copy.deepcopy(annotations)
    else:
      if self.config.duplicate_results:
        annotations.extend(copy.deepcopy(self.last_annotations))
        for annotation in annotations:
          annotation.timestamp = timestamp
      else:
        annotations.append(None)

    return True
