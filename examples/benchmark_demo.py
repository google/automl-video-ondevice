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
"""Test out a model.

For Jetson devices, you must specify a .pb model:
  python3 examples/benchmark_demo.py --model=data/traffic_model_tftrt.pb

For Coral devices, you must specify a tflite or _edgetpu.tflite model:
  python3 examples/benchmark_demo.py --model=data/traffic_model_edgetpu.tflite

"""
import argparse
import time
import numpy as np

from automl_video_ondevice import object_tracking as vot
import utils

default_model = 'data/traffic_model_edgetpu.tflite'
default_labels = 'data/traffic_label_map.pbtxt'


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', help='model path', default=default_model)
  parser.add_argument(
      '--labels', help='label file path', default=default_labels)
  parser.add_argument(
      '--threshold', type=float, default=0.2, help='class score threshold')
  args = parser.parse_args()

  print('Loading %s with %s labels.' % (args.model, args.labels))

  config = vot.ObjectTrackingConfig(score_threshold=args.threshold)
  engine = vot.load(args.model, args.labels, config)
  input_size = engine.input_size()
  fps_calculator = utils.FpsCalculator()

  blank_image = np.zeros((input_size.height, input_size.width, 3),
                         dtype=np.uint8)

  while True:
    # Run inference engine to populate annotations array.
    annotations = []
    timestamp = int(round(time.time() * 1000))
    engine.run(timestamp, blank_image, annotations)

    # Calculate FPS and latency.
    fps, latency = fps_calculator.measure()
    print('FPS: {}\t\t\tLatency: {}ms'.format(fps, latency))


if __name__ == '__main__':
  main()
