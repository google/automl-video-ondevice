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

python3 examples/jetson_camera_demo \
  --model data/traffic_model_edgetpu.tflite \
  --label data/traffic_label_map.pbtxt

Press Q key to exit.
"""
import argparse
import time
from automl_video_ondevice import object_tracking as vot
import utils

try:
  import cv2  
except:  
  print("Couldn't load cv2. Try running: sudo apt install python3-opencv.")

current_milli_time = lambda: int(round(time.time() * 1000))


def main():
  default_model = 'data/traffic_model_tftrt.pb'
  default_labels = 'data/traffic_label_map.pbtxt'
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', help='model path', default=default_model)
  parser.add_argument(
      '--labels', help='label file path', default=default_labels)
  parser.add_argument(
      '--threshold', type=float, default=0.25, help='class score threshold')
  parser.add_argument(
      '--use_tracker', type=bool, default=False, help='use an object tracker')
  parser.add_argument(
      '--video_device',
      help='-1 for ribbon-cable camera. >= 0 for USB camera. '
      'If both are plugged in, the USB camera will have the ID "1".',
      type=int,
      default=-1)
  parser.add_argument(
      '--video_width', help='Input video width.', type=int, default=1280)
  parser.add_argument(
      '--video_height', help='Input video height.', type=int, default=720)
  args = parser.parse_args()

  print('Loading %s with %s labels.' % (args.model, args.labels))

  config = vot.ObjectTrackingConfig(
      score_threshold=args.threshold,
      tracker=vot.Tracker.FAST_INACCURATE
      if args.use_tracker else vot.Tracker.NONE)
  engine = vot.load(args.model, args.labels, config)
  input_size = engine.input_size()
  fps_calculator = utils.FpsCalculator()

  if args.video_device >= 0:
    cap = cv2.VideoCapture(
        'v4l2src device=/dev/video{} ! videoconvert ! '
        'videoscale method=0 add-borders=false ! '
        'video/x-raw, width={}, height={}, format=RGB ! videoconvert ! '
        'appsink'.format(args.video_device, args.video_width,
                         args.video_height), cv2.CAP_GSTREAMER)
  else:
    cap = cv2.VideoCapture(
        'nvarguscamerasrc ! nvvidconv ! '
        'video/x-raw, format=(string)BGRx ! videoconvert ! '
        'videoscale method=0 add-borders=false ! '
        'video/x-raw, width={}, height={}, format=RGB ! videoconvert ! '
        'appsink'.format(args.video_width, args.video_height),
        cv2.CAP_GSTREAMER)

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    # Resizes frame.
    resized_frame = cv2.resize(frame, (input_size.width, input_size.height))

    # Grabs current millisecond for timestamp.
    timestamp = current_milli_time()

    # Run inference engine to populate annotations array.
    annotations = []
    if engine.run(timestamp, resized_frame, annotations):
      frame = utils.render_bbox(frame, annotations)

    # Calculate FPS, then visualize it.
    fps, latency = fps_calculator.measure()
    frame = cv2.putText(frame, '{} fps'.format(fps), (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    frame = cv2.putText(frame, '{} ms'.format(latency), (0, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
