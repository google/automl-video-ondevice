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
"""Loops through the images in the data folder and prints the bounding box output.

For Jetson devices, you must specify a .pb model:
  python3 examples/image_demo.py --model=data/traffic_model_tftrt.pb

For Coral devices, you must specify a tflite or _edgetpu.tflite model:
  python3 examples/image_demo.py --model=data/traffic_model_edgetpu.tflite
"""

import argparse
from PIL import Image
from automl_video_ondevice import object_tracking as vot

default_model = 'data/traffic_model_edgetpu.tflite'
default_labels = 'data/traffic_label_map.pbtxt'

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', help='model path', default=default_model)
  parser.add_argument(
      '--labels', help='label file path', default=default_labels)
  parser.add_argument(
      '--threshold', type=float, default=0.2, help='class score threshold')
  args = parser.parse_args()

  config = vot.ObjectTrackingConfig(score_threshold=args.threshold)
  engine = vot.load(args.model, args.labels, config)
  input_size = engine.input_size()

  for i in range(1, 9):
    image = Image.open('data/traffic_frames/000%d.bmp' %
                       i).convert('RGB').resize(
                           (input_size.width, input_size.height))

    out = []
    if engine.run(1, image, out):
      for annotation in out:
        print('{}: {} [{}, {}, {}, {}]'.format(annotation.class_name,
                                               annotation.confidence_score,
                                               annotation.bbox.top,
                                               annotation.bbox.left,
                                               annotation.bbox.bottom,
                                               annotation.bbox.right))
    else:
      print('Couldn\'t run inferencing!')
