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
"""Provides an implementation of object tracking using TF and TF-TRT."""

import cv2
import numpy as np
from automl_video_ondevice.object_tracking.base_object_detection import BaseObjectDetectionInference


class SingleTracker:
  """A single tracklet."""

  def __init__(self, term_crit, init_box, frame, tracker_id, annotation):
    (x, y, x2, y2) = init_box
    self.height, self.width = frame.shape[:2]
    self.term_crit = term_crit
    self.box = self.relative_to_glob((x, y, x2 - x, y2 - y))
    # self.calculate_roi_hist(frame)
    self.tracker_id = tracker_id
    self.corrected = True
    self.annotation = annotation
    self.annotation.track_id = tracker_id
    self.age = 0
    self.reset_health()
    self.create_kalman()
    self.correct(init_box, frame)

  def create_kalman(self):
    """Creates kalman filter."""

    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]],
                                        np.float32)

    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
                                       [0, 0, 0, 1]], np.float32) * 0.03
    self.kalman = kalman
    self.measurement = np.array((2, 1), np.float32)
    self.prediction = np.zeros((2, 1), np.float32)

  def relative_to_glob(self, box):
    """Converts space to [0-1, 0-1] -> [0-width, 0-height]."""
    (x, y, x2, y2) = box
    return (np.clip(int(x * self.width), 0,
                    self.width), np.clip(int(y * self.height), 0, self.height),
            np.clip(int(x2 * self.width), 0,
                    self.width), np.clip(int(y2 * self.height), 0, self.height))

  def glob_to_relative(self, box):
    (x, y, x2, y2) = box
    return (x / self.width, y / self.height, x2 / self.width, y2 / self.height)

  # Run this only a couple times.
  def calculate_roi_hist(self, frame):
    """Calculates region of interest histogram.

    Args:
      frame: The np.array image frame to calculate ROI histogram for.
    """
    (x, y, w, h) = self.box
    roi = frame[y:y + h, x:x + w]

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                       np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask, [180, 255],
                            [0, 180, 0, 255])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    self.roi_hist = roi_hist

  # Run this every frame
  def run(self, frame):
    """Processes a single frame.

    Args:
      frame: The np.array image frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], self.roi_hist, [0, 180, 0, 255], 1)
    _, self.box = cv2.CamShift(dst, self.box, self.term_crit)

    (x, y, x2, y2) = self.glob_to_relative(
        (self.box[0], self.box[1], self.box[0] + self.box[2],
         self.box[1] + self.box[3]))

    self.annotation.bbox.left = x
    self.annotation.bbox.top = y
    self.annotation.bbox.right = x2
    self.annotation.bbox.bottom = y2

    self.age = self.age + 1
    self.degrade()

  def correct(self, new_box, frame):
    """Corrects current tracklet with new information.

    Args:
      new_box: incoming bounding boxes.
      frame: The np.array image frame.
    """
    (x, y, x2, y2) = new_box
    self.box = self.relative_to_glob((x, y, x2 - x, y2 - y))
    self.calculate_roi_hist(frame)

    self.annotation.bbox.left = x
    self.annotation.bbox.top = y
    self.annotation.bbox.right = x2
    self.annotation.bbox.bottom = y2

    center_point = np.array([
        np.float32(self.box[0] + self.box[2] * 0.5),
        np.float32(self.box[1] + self.box[3] * 0.5)
    ], np.float32)
    self.kalman.correct(center_point)

    self.reset_health()

  def get_current_box(self):
    (x, y, w, h) = self.box
    return self.glob_to_relative((x, y, x + w, y + h))

  def reset_health(self):
    self.health = 10

  def degrade(self):
    self.health = self.health - 1


class TrackerEngine:
  """Camshift-based tracker for use on top of object detection."""

  def __init__(self):
    self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    self.tracks = []
    self.current_track = 0

  def predict(self, frame, predictions):
    """Creates new predictions for frames with missed detections.

    Args:
      frame: frame data to aid with prediction.
      predictions: output prediction array.
    """
    new_tracks = []
    for track in self.tracks:
      if track.health <= 0:
        continue

      track.run(frame)
      predictions.append(track.annotation)
      for other_track in self.tracks:
        if track is other_track:
          continue
        iou = get_iou(other_track.get_current_box(), track.get_current_box())
        if iou > 0.1:
          if track.age > other_track.age:
            other_track.health = -1
          else:
            track.health = -1

    for track in self.tracks:
      if track.health > 0:
        new_tracks.append(track)
    self.tracks = new_tracks

  def correct(self, annotations, frame):
    """Corrects the current tracker pipeline with new annotations.

    Correction can involve adding new boxes, or re-fitting out of sync boxes.

    Ideally you want to be running inferencing on a separate thread,
    and when the inferencing is done you can correct the tracker with the new
    boxes.

    Args:
      annotations: input annotations to correct the system with.
      frame: frame data to aid with the correction.
    """
    for track in self.tracks:
      track.corrected = False

    for annotation in annotations:
      annotation_bbox = (annotation.bbox.left, annotation.bbox.top,
                         annotation.bbox.right, annotation.bbox.bottom)

      corrected_track = False
      for track in self.tracks:
        if track.corrected:
          continue

        iou = get_iou(annotation_bbox, track.get_current_box())
        if iou > 0.1:
          track.correct(annotation_bbox, frame)
          track.corrected = True
          corrected_track = True
          break

      if not corrected_track:
        self.tracks.append(
            SingleTracker(self.term_crit, annotation_bbox, frame,
                          self.current_track, annotation))
        self.current_track = self.current_track + 1


def get_iou(bb1, bb2):
  """Calculate the Intersection over Union (IoU) of two bounding boxes.

  Args:
    bb1: dict {'x1', 'x2', 'y1', 'y2'} The (x1, y1) position is at the top left
      corner, the (x2, y2) position is at the bottom right corner
    bb2: dict {'x1', 'x2', 'y1', 'y2'} The (x, y) position is at the top left
      corner, the (x2, y2) position is at the bottom right corner

  Returns:
    float in [0, 1]
  """

  # determine the coordinates of the intersection rectangle
  (bb1_x1, bb1_y1, bb1_x2, bb1_y2) = bb1
  (bb2_x1, bb2_y1, bb2_x2, bb2_y2) = bb2

  x_left = max(bb1_x1, bb2_x1)
  y_top = max(bb1_y1, bb2_y1)
  x_right = min(bb1_x2, bb2_x2)
  y_bottom = min(bb1_y2, bb2_y2)

  if x_right < x_left or y_bottom < y_top:
    return 0.0

  # The intersection of two axis-aligned bounding boxes is always an
  # axis-aligned bounding box
  intersection_area = (x_right - x_left) * (y_bottom - y_top)

  # compute the area of both AABBs
  bb1_area = (bb1_x2 - bb1_x1) * (bb1_y2 - bb1_y1)
  bb2_area = (bb2_x2 - bb2_x1) * (bb2_y2 - bb2_y1)

  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
  assert iou >= 0.0
  assert iou <= 1.0
  return iou


class CamshiftObjectTracker(BaseObjectDetectionInference):
  """Camshift and Kalman Filter-based tracking."""

  def __init__(self, object_detection_engine, config):
    del config  # Not used, yet.
    self._tracker_engine = TrackerEngine()
    self._object_detection_engine = object_detection_engine

  def input_size(self):
    return self._object_detection_engine.input_size()

  def run(self, timestamp, frame, annotations):
    np_frame = np.array(frame)
    detection_annotations = []
    if self._object_detection_engine.run(timestamp, np_frame,
                                         detection_annotations):
      self._tracker_engine.predict(np_frame, annotations)
      self._tracker_engine.correct(detection_annotations, np_frame)
      return True
    else:
      return False
