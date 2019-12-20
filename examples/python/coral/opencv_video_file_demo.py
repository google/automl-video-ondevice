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

python3 opencv_video_file_demo \
  --input_video data/traffic_frames.mp4
  --output_video data/traffic_frames_annotated.mp4
  --model data/traffic_model_edgetpu.tflite \
  --label data/traffic_label_map.pbtxt

To output to UI instead of file, do not include the "--output_video" argument.

python3 opencv_video_file_demo \
  --input_video data/traffic_frames.mp4
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
  parser.add_argument('--input_video', help='input video file path', default='')
  parser.add_argument(
      '--output_video', help='output video file path', default='')
  parser.add_argument(
      '--threshold', type=float, default=0.2, help='class score threshold')
  args = parser.parse_args()

  print('Loading %s with %s labels.' % (args.model, args.labels))

  config = automl_ondevice.ObjectTrackingConfig(score_threshold=args.threshold)
  engine = automl_ondevice.ObjectTrackingInference.TFLiteModel(
      args.model, args.labels, config)
  input_size = engine.getInputSize()

  cap = cv2.VideoCapture(args.input_video)

  if cap.isOpened() and args.output_video:
    print(args.output_video, int(cap.get(cv2.CAP_PROP_FOURCC)),
          cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    writer = cv2.VideoWriter(args.output_video,
                             int(cap.get(cv2.CAP_PROP_FOURCC)),
                             cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
  else:
    writer = None

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

    if writer:
      writer.write(cv2_im)
    else:
      cv2.imshow('frame', cv2_im)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  writer.release()
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
