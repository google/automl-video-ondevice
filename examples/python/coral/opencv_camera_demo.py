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
r"""A demo which runs object detection on camera frames.

Requires cv2 from `sudo apt-get install python3-opencv`

python3 opencv_camera_demo \
  --model data/traffic_model_edgetpu.tflite \
  --label data/traffic_label_map.pbtxt

Press Q key to exit.
"""
import argparse
import time
import automl_ondevice
import cv2
from PIL import Image

current_milli_time = lambda: int(round(time.time() * 1000))


def main():
  default_model = 'data/traffic_model_edgetpu.tflite'
  default_labels = 'data/traffic_label_map.pbtxt'
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', help='.tflite model path', default=default_model)
  parser.add_argument(
      '--labels', help='label file path', default=default_labels)
  parser.add_argument(
      '--threshold', type=float, default=0.2, help='class score threshold')
  args = parser.parse_args()

  print('Loading %s with %s labels.' % (args.model, args.labels))

  config = automl_ondevice.ObjectTrackingConfig(score_threshold=args.threshold)
  engine = automl_ondevice.ObjectTrackingInference.TFLiteModel(
      args.model, args.labels, config)
  input_size = engine.getInputSize()
  fps_time_queue = []  # Used to calculate FPS.

  cap = cv2.VideoCapture(0)

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    # Converts image to PIL Image.
    cv2_im = frame
    pil_im = Image.fromarray(cv2_im).convert('RGB').resize(
        (input_size.width, input_size.height))

    # Grabs current millisecond for timestamp.
    timestamp = current_milli_time()

    # Run inference engine to populate annotations array.
    annotations = []
    if engine.run(timestamp, pil_im, annotations):
      cv2_im = append_objs_to_img(cv2_im, annotations)

    # Calculate FPS based on sample size of 15.
    fps_time_queue.append(timestamp)
    if len(fps_time_queue) > 15:
      fps_time_queue.pop(0)
    if len(fps_time_queue) > 2:
      fps = round(
          len(fps_time_queue) / (fps_time_queue[-1] - fps_time_queue[0]) * 1000,
          2)
      latency = round(fps_time_queue[-1] - fps_time_queue[-2], 4)
    else:
      fps = 'N/A'
      latency = 'N/A'

    cv2_im = cv2.putText(cv2_im, '{} fps'.format(fps), (0, 20),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2_im = cv2.putText(cv2_im, '{} ms'.format(latency), (0, 40),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('frame', cv2_im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


def append_objs_to_img(cv2_im, annotations):
  height, width, _ = cv2_im.shape
  for annotation in annotations:
    x0, y0, x1, y1 = (annotation.bbox.left, annotation.bbox.top,
                      annotation.bbox.right, annotation.bbox.bottom)
    # Converts coordinates from relative [0,1] space to [width,height]
    x0, y0, x1, y1 = int(x0 * width), int(y0 * height), int(x1 * width), int(
        y1 * height)
    percent = int(100 * annotation.confidence_score)
    label = '%d%% %s' % (percent, annotation.class_name)

    cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2_im = cv2.putText(cv2_im, label, (x0, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                         1.0, (255, 0, 0), 2)
  return cv2_im


if __name__ == '__main__':
  main()
