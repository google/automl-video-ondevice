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
r"""Compiles an AutoML Video frozen graph for TensorTRT inferencing.

Needs to be copied to and ran on-device.

python3 trt_compiler \
  --model data/traffic_model.pb
"""
import argparse
import os
import tensorflow.compat.v1 as tf
import tensorflow.contrib.tensorrt as trt


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', help='.pb model path')
  parser.add_argument(
      '--downgrade',
      help='Downgrades the model for use with Tensorflow 1.14 '
      '(There maybe some quality degradation.)',
      action='store_true')
  args = parser.parse_args()

  filename, extension = os.path.splitext(args.model)
  output_file_path = '{}_trt{}'.format(filename, extension)

  frozen_graph = tf.GraphDef()
  with open(args.model, 'rb') as f:
    frozen_graph.ParseFromString(f.read())

  if args.downgrade:
    downgrade_equal_op(frozen_graph)
    downgrade_nmv5_op(frozen_graph)
    downgrade_fused_batch_norm_v3_op(frozen_graph)
    downgrade_depthwise_conv2d_native_op(frozen_graph)

  is_lstm = check_lstm(frozen_graph)
  if is_lstm:
    print('Converting LSTM model.')

  trt_graph = trt.create_inference_graph(
      input_graph_def=frozen_graph,
      outputs=[
          'detection_boxes', 'detection_classes', 'detection_scores',
          'num_detections'
      ] + ([
          'raw_outputs/lstm_c', 'raw_outputs/lstm_h', 'raw_inputs/init_lstm_c',
          'raw_inputs/init_lstm_h'
      ] if is_lstm else []),
      max_batch_size=1,
      max_workspace_size_bytes=1 << 25,
      precision_mode='FP16',
      minimum_segment_size=50)

  with open(output_file_path, 'wb') as f:
    f.write(trt_graph.SerializeToString())


def check_lstm(graph_def):
  for n in graph_def.node:
    if n.name in [
        'raw_outputs/lstm_c', 'raw_outputs/lstm_h', 'raw_inputs/init_lstm_c',
        'raw_inputs/init_lstm_h'
    ]:
      return True
  return False


def downgrade_equal_op(graph_def):
  for n in graph_def.node:
    if n.op == 'Equal':
      del n.attr['incompatible_shape_error']


def downgrade_fused_batch_norm_v3_op(graph_def):
  for n in graph_def.node:
    if n.op == 'FusedBatchNormV3':
      del n.attr['exponential_avg_factor']


def downgrade_depthwise_conv2d_native_op(graph_def):
  for n in graph_def.node:
    if n.op == 'DepthwiseConv2dNative':
      del n.attr['explicit_paddings']


def downgrade_nmv5_op(graph_def):
  nms_names = []
  nms_score = []
  nms_toreplace = []

  # Recreates NMSV5's selected score output.
  def score_mapper(graph_def, output_name, input_score, input_index):
    graph = tf.Graph()
    with graph.as_default():
      tf_input_score = tf.placeholder(tf.float32, [1], name='tmp/input_score')
      tf_input_index = tf.placeholder(tf.int32, [1], name='tmp/input_index')
      tf.gather(tf_input_score, tf_input_index, name=output_name)

    tmp_graph_def = graph.as_graph_def()
    for node in tmp_graph_def.node:
      if node.name == 'tmp/input_score':
        tmp_graph_def.node.remove(node)
    for node in tmp_graph_def.node:
      if node.name == 'tmp/input_index':
        tmp_graph_def.node.remove(node)
      for i in range(len(node.input)):
        if node.input[i] == 'tmp/input_score':
          node.input[i] = input_score
        if node.input[i] == 'tmp/input_index':
          node.input[i] = input_index
    graph_def.node.extend(tmp_graph_def.node)

  # First pass; adds a selected_score output to every NMSV5 operation.
  for n in graph_def.node:
    if n.op == 'NonMaxSuppressionV5':
      nms_names.append('%s:1' % n.name)
      nms_score.append('%s/selected_score:0' % n.name)
      nms_toreplace.append('%s:2' % n.name)
      score_mapper(graph_def, '%s/selected_score' % n.name, n.input[1],
                   '%s:0' % n.name)

  # Second pass; rearranges output order, deletes original selected_score output
  # from the node.
  for n in graph_def.node:
    if n.op == 'NonMaxSuppressionV5':
      n.op = 'NonMaxSuppressionV4'
      del n.input[-1]

    for i in range(len(n.input)):
      if n.input[i] in nms_names:
        n.input[i] = nms_score[nms_names.index(n.input[i])]
      if n.input[i] in nms_toreplace:
        n.input[i] = nms_names[nms_toreplace.index(n.input[i])]


if __name__ == '__main__':
  main()
