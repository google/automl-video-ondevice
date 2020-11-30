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
from typing import List
from typing import Union
import numpy as np
import tensorflow.compat.v1 as tf
from automl_video_ondevice.shot_classification.base_shot_classification import BaseShotClassificationInference
from automl_video_ondevice.shot_classification.config import ShotClassificationConfig
from automl_video_ondevice.types import ShotClassificationAnnotation
from automl_video_ondevice.types import Size

import automl_video_ondevice.utils as vot_utils
import tensorflow.contrib.tensorrt as trt  # pylint: disable=g-explicit-tensorflow-version-import,unused-import


class TFShotClassificationInference(BaseShotClassificationInference):
  """Implementation of the BaseShotClassificationInference using TF, or TF-TRT.

  TF-TRT is a hybrid solution, converting a sub-section of the Tensorflow graph
  into it's proprietary TensorRT format. This allows for maximum usage of
  NVIDIA devices while keeping full Tensorflow support.

  Non-TRT frozen graphs can also be used with this implementation.
  """

  def __init__(self, frozen_graph_path: str, label_map_path: str,
               config: ShotClassificationConfig):
    """Constructor for TFShotClassificationInference.

    Args:
      frozen_graph_path: String value for the file path of frozen graph.
      label_map_path: String value for the file path of the label map.
      config: ShotClassificationConfig object with shot classification configs.
    """
    self.config = config

    if self.config.top_k <= 0:
      raise ValueError(
          'Top k cannot be zero, or else no results will be returned.')

    self._load_label_map(label_map_path)
    self._load_frozen_graph(frozen_graph_path)

  def __del__(self):
    """Destructor for TFShotClassificationInference."""
    self.session.close()

  def _load_label_map(self, label_map_path: str):
    """Opens and parses a given shot classification label map into memory.

    The standard AutoML Video .pbtxt label map is assumed:
      item {name: "Cat" id: 0}
      item {name: "Dog" id: 1}

    Args:
      label_map_path: String value for the file path of the label map.
    """
    with open(label_map_path, 'r') as f:
      self.label_map, _ = vot_utils.parse_label_map(f.read())

  def _load_frozen_graph(self, frozen_graph_path: str):
    """Opens and loads a frozen graph into memory.

    Args:
      frozen_graph_path: String value for the file path of frozen graph.
    """
    frozen_graph = tf.GraphDef()
    with open(frozen_graph_path, 'rb') as f:
      frozen_graph.ParseFromString(f.read())

    self._is_lstm = self._check_lstm(frozen_graph)
    if self._is_lstm:
      print('Loading an LSTM model.')

    self.graph = tf.Graph()
    with self.graph.as_default():
      self.output_node = tf.import_graph_def(
          frozen_graph,
          return_elements=[
              'probabilities:0',
          ] + (['raw_outputs/lstm_c:0', 'raw_outputs/lstm_h:0']
               if self._is_lstm else []))
    self.session = tf.InteractiveSession(graph=self.graph)

    tf_probabilities = self.graph.get_tensor_by_name('import/probabilities:0')
    if self._is_lstm:
      tf_lstm_c = self.graph.get_tensor_by_name('import/raw_outputs/lstm_c:0')
      tf_lstm_h = self.graph.get_tensor_by_name('import/raw_outputs/lstm_h:0')

    self._output_nodes = [tf_probabilities
                         ] + ([tf_lstm_c, tf_lstm_h] if self._is_lstm else [])

    if self._is_lstm:
      self.lstm_c = np.ones(tf_lstm_c.get_shape())
      self.lstm_h = np.ones(tf_lstm_h.get_shape())

    self.sliding_window = None
    self.frames_since_last_inference = self.config.inference_rate
    self.last_annotations = []

  def _check_lstm(self, graph_def: tf.GraphDef) -> bool:
    """Checks to see if the input frozen graph is an LSTM graph.

    LSTM is determined by the existence of the raw_outputs/lstm_c and
    raw_outputs/lstm_h nodes. Some informational tensorflow node may be used in
    the future for determining the model type.

    Args:
      graph_def: Loaded frozen graph. This should not be a loaded tensorflow
        session, since LSTM needs to be determined before the model is loaded.
    Returns:
      If the frozen graph is of an LSTM model.
    """
    for node in graph_def.node:
      if node.name == 'raw_outputs/lstm_c' or node.name == 'raw_outputs/lstm_h':
        return True
    return False

  def input_size(self) -> Size:
    """Calculate / grab optimal input size.

    The user is expected to ensure the size of their input image is correct.
    This is in case the user wants to do any acceleration of image resizing
    themselves. Although this model accepts any input size, it is currently
    recommended that the input is the same as the size used in training.

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
    np_frame = np.array(frame)

    # User has to handle frame resizing by themselves.
    if np.shape(np_frame)[0] != 256:
      raise ValueError(
          'Shot classification input must be a height of 256 pixels. '
          'There is no width limit. Aspect ratio must be retained.')

    if self.config.sliding_window_size == 1 or self._is_lstm:
      # A sliding window size 1 is just the current frame. It still needs to be
      # expanded for the input batch size.
      #
      # LSTM is forced to have a sliding window size 1.
      self.sliding_window = np.expand_dims(np_frame, axis=0)
    else:
      # Initiate sliding window if not created yet.
      if self.sliding_window is None:
        self.sliding_window = np.zeros(
            (self.config.sliding_window_size, np.shape(np_frame)[0],
             np.shape(np_frame)[1], 3))

      # Moves sliding window and adds new frame to the back
      # No need to move the sliding window if window size is only 1.
      self.sliding_window = np.roll(self.sliding_window, -1)
      self.sliding_window[self.config.sliding_window_size - 1] = np_frame

    self.frames_since_last_inference += 1
    if self.frames_since_last_inference >= self.config.inference_rate or self._is_lstm:
      self.frames_since_last_inference = 0
      with self.graph.as_default():
        # Tensors to feed in.
        feed_dict = {
            'import/video_inputs:0': self.sliding_window,
        }
        if self._is_lstm:
          feed_dict.update({
              'import/raw_inputs/init_lstm_c:0': self.lstm_c,
              'import/raw_inputs/init_lstm_h:0': self.lstm_h
          })

        session_return = self.session.run(
            self._output_nodes, feed_dict=feed_dict)
        if self._is_lstm:
          (probabilities, self.lstm_c, self.lstm_h) = session_return
          score = probabilities
        else:
          (probabilities) = session_return
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
