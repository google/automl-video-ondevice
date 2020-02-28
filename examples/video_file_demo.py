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
r"""Runs object detection on video files.

Can output to either a video file or to the UI (default).

Requires cv2 from `sudo apt-get install python3-opencv`

Note: this specific example uses the EdgeTPU .tflite model. To run inferencing
on a jetson nano, a .pb file must be passed to the --model argument instead.

For Jetson Nano:
  python3 examples/video_file_demo.py \
    --input_video data/traffic_frames.mp4
    --output_video data/traffic_frames_annotated.mp4
    --model data/traffic_model_tftrt.pb \
    --label data/traffic_label_map.pbtxt

For Coral Devices:
  python3 examples/video_file_demo.py \
    --input_video data/traffic_frames.mp4
    --output_video data/traffic_frames_annotated.mp4
    --model data/traffic_model_edgetpu.tflite \
    --label data/traffic_label_map.pbtxt

To output to UI instead of file, do not include the "--output_video" argument.

python3 examples/video_file_demo.py \
  --input_video data/traffic_frames.mp4
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
  default_video = 'data/traffic_frames.mp4'
  default_model = 'data/traffic_model_edgetpu.tflite'
  default_labels = 'data/traffic_label_map.pbtxt'
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', help='model path', default=default_model)
  parser.add_argument(
      '--labels', help='label file path', default=default_labels)
  parser.add_argument(
      '--input_video', help='input video file path', default=default_video)
  parser.add_argument(
      '--output_video', help='output video file path', default='')
  parser.add_argument(
      '--threshold', type=float, default=0.2, help='class score threshold')
  args = parser.parse_args()

  print('Loading %s with %s labels.' % (args.model, args.labels))

  config = vot.ObjectTrackingConfig(score_threshold=args.threshold)
  engine = vot.load(args.model, args.labels, config)
  input_size = engine.input_size()

  cap = cv2.VideoCapture(args.input_video)

  writer = None
  if cap.isOpened() and args.output_video:
    writer = cv2.VideoWriter(args.output_video,
                             int(cap.get(cv2.CAP_PROP_FOURCC)),
                             cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

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

    if writer:
      writer.write(frame)
    else:
      cv2.imshow('frame', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  if writer:
    writer.release()
  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
