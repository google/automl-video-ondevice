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
r"""Compiles an AutoML Video frozen graph for TensorTRT inferencing.

Needs to be copied to and ran on-device.

python3 trt_compiler \
  --model data/traffic_model.pb
"""
import argparse
import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', help='.pb model path')
  args = parser.parse_args()

  filename, extension = os.path.splitext(args.model)
  output_file_path = '{}_trt{}'.format(filename, extension)

  frozen_graph = tf.GraphDef()
  with open(args.model, 'rb') as f:
    frozen_graph.ParseFromString(f.read())

  trt_graph = trt.create_inference_graph(
      input_graph_def=frozen_graph,
      outputs=[
          'detection_boxes', 'detection_classes', 'detection_scores',
          'num_detections', 'raw_outputs/lstm_c', 'raw_outputs/lstm_h',
          'raw_inputs/init_lstm_c', 'raw_inputs/init_lstm_h'
      ],
      max_batch_size=1,
      max_workspace_size_bytes=1 << 25,
      precision_mode='FP16',
      minimum_segment_size=50)

  with open(output_file_path, 'wb') as f:
    f.write(trt_graph.SerializeToString())


if __name__ == '__main__':
  main()
