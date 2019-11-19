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
"""
Before running, please copy libedgetpu.so and automl_ondevice.so to the same
folder as this python script.

cp ../../third_party/edgetpu/libedgetpu/direct/k8/libedgetpu.so.1 .
cp ../../bin/k8/py/automl_ondevice.so .
python3 ondevice_demo.py
"""

import automl_ondevice
from PIL import Image

if __name__ == '__main__':
  config = automl_ondevice.ObjectTrackingConfig(score_threshold=0.2)
  engine = automl_ondevice.ObjectTrackingInference.TFLiteModel(
      '../../data/traffic_model.tflite', '../../data/traffic_label_map.pbtxt',
      config)
  input_size = engine.getInputSize()

  for i in range(1, 9):
    image = Image.open('../../data/traffic_frames/000%d.bmp' %
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
