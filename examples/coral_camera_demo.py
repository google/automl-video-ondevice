# Lint as: python3
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
# ==============================================================================
r"""A demo which runs object detection on camera frames.

Requires cv2 from `sudo apt-get install python3-opencv`

python3 examples/coral_camera_demo \
  --model data/traffic_model_edgetpu.tflite \
  --label data/traffic_label_map.pbtxt

Press Q key to exit.
"""
import argparse
import time
from PIL import Image
from automl_video_ondevice import object_tracking as vot
import utils

try:
  import cv2  
except:  
  print("Couldn't load cv2. Try running: sudo apt install python3-opencv.")

current_milli_time = lambda: int(round(time.time() * 1000))
default_model = 'data/traffic_model_edgetpu.tflite'
default_labels = 'data/traffic_label_map.pbtxt'


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='model path', default=default_model)
  parser.add_argument(
      '--labels', help='label file path', default=default_labels)
  parser.add_argument(
      '--threshold', type=float, default=0.2, help='class score threshold')
  parser.add_argument(
      '--use_tracker', type=bool, default=False, help='use an object tracker')
  args = parser.parse_args()

  print('Loading %s with %s labels.' % (args.model, args.labels))

  config = vot.ObjectTrackingConfig(
      score_threshold=args.threshold,
      tracker=vot.Tracker.FAST_INACCURATE
      if args.use_tracker else vot.Tracker.NONE)
  engine = vot.load(args.model, args.labels, config)
  input_size = engine.input_size()
  fps_calculator = utils.FpsCalculator()

  cap = cv2.VideoCapture(0)

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    # Converts image to PIL Image.
    pil_im = Image.fromarray(frame).convert('RGB').resize(
        (input_size.width, input_size.height))

    # Grabs current millisecond for timestamp.
    timestamp = current_milli_time()

    # Run inference engine to populate annotations array.
    annotations = []
    if engine.run(timestamp, pil_im, annotations):
      frame = utils.render_bbox(frame, annotations)

    # Calculate FPS, then visualize it.
    fps, latency = fps_calculator.measure()
    frame = cv2.putText(frame, '{} fps'.format(fps), (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    frame = cv2.putText(frame, '{} ms'.format(latency), (0, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
