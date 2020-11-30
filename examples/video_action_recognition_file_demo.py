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
r"""Runs action recognition on video files.

Can output to either a video file or to the UI (default).

Requires cv2 from `sudo apt-get install python3-opencv`

python3 examples/video_file_demo.py \
  --input_video data/action_recognition.mp4
  --input_video data/action_recognition_annotated.mp4
  --model data/action_recognition_model.pb \
  --label data/action_recognition_label_map.pbtxt

To output to UI instead of file, do not include the "--output_video" argument.

python3 examples/video_file_demo.py \
  --input_video data/action_recognition.mp4
  --model data/action_recognition_model.pb \
  --label data/action_recognition_label_map.pbtxt

Press Q key to exit.
"""
import argparse
from automl_video_ondevice import action_recognition as var
import utils

try:
  import cv2  # pylint: disable=g-import-not-at-top
except:  # pylint: disable=bare-except
  print("Couldn't load cv2. Try running: sudo apt install python3-opencv.")


def main():
  default_video = 'data/action_recognition.mp4'
  default_model = 'data/action_recognition_model.pb'
  default_labels = 'data/action_recognition_label_map.pbtxt'
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
  parser.add_argument(
      '--use_tracker', type=bool, default=False, help='use an object tracker')
  parser.add_argument(
      '--top_k',
      type=int,
      default=1,
      help='The number of results to return, ordered by highest to lowest score.'
  )
  args = parser.parse_args()

  print('Loading %s with %s labels.' % (args.model, args.labels))

  config = var.ActionRecognitionConfig(
      score_threshold=args.threshold, top_k=args.top_k)
  engine = var.load(args.model, args.labels, config)
  input_size = engine.input_size()

  cap = cv2.VideoCapture(args.input_video)

  writer = None
  if cap.isOpened() and args.output_video:
    writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'),
                             cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

  timestamp = 0
  while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
      break

    # Resizes frame.
    resized_frame = cv2.resize(frame, (input_size.width, input_size.height))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Calculates current microsecond for timestamp.
    timestamp = int(timestamp + (1/cap.get(cv2.CAP_PROP_FPS)) * 1000 * 1000)

    # Run inference engine to populate annotations array.
    annotations = []
    if engine.run(timestamp, rgb_frame, annotations):
      frame = utils.render_classifications(frame, annotations)

    if writer:
      writer.write(frame)
    else:
      cv2.imshow('frame', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  if writer:
    writer.release()
  else:
    cv2.destroyAllWindows()
  cap.release()


if __name__ == '__main__':
  main()
