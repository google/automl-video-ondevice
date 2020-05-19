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
"""Provides an implementation of object tracking using TF and TF-TRT."""

import numpy as np
from automl_video_ondevice.object_tracking.base_object_detection import BaseObjectDetectionInference
from automl_video_ondevice.types import NormalizedBoundingBox
from automl_video_ondevice.types import ObjectTrackingAnnotation

try:
  import platform
  if platform.machine().lower() == 'aarch64':
    from automl_video_ondevice.object_tracking.mediapipe_tracker.aarch64 import mediapipe_tracker  
  else:
    raise ImportError('Unsupported architecture.')
except ImportError:
  raise RuntimeError(
      'The basic tracker is not publicly available yet, '
      'please reach out to repo maintainers for early access. '
      'If this error persists, it may mean your CPU architecture is not yet '
      'supported.')

mediapipe_graph = """
input_stream: "input_frame"
input_stream: "input_detections"
output_stream: "output_tracked"

# Assigns an unique id for each new detection.
node {
  calculator: "DetectionUniqueIdCalculator"
  input_stream: "DETECTIONS:input_detections"
  output_stream: "DETECTIONS:detections_with_id"
}

# Converts detections to TimedBox protos which are used as initial location
# for tracking.
node {
  calculator: "DetectionsToTimedBoxListCalculator"
  input_stream: "DETECTIONS:detections_with_id"
  output_stream: "BOXES:start_pos"
}


node: {
  calculator: "ImageTransformationCalculator"
  input_stream: "IMAGE:input_frame"
  output_stream: "IMAGE:downscaled_input_video"
  node_options: {
    [type.googleapis.com/mediapipe.ImageTransformationCalculatorOptions] {
      output_width: 640
      output_height: 360
    }
  }
}

# Performs motion analysis on an incoming video stream.
node: {
  calculator: "MotionAnalysisCalculator"
  input_stream: "VIDEO:downscaled_input_video"
  output_stream: "CAMERA:camera_motion"
  output_stream: "FLOW:region_flow"

  node_options: {
    [type.googleapis.com/mediapipe.MotionAnalysisCalculatorOptions]: {
      analysis_options {
        analysis_policy: ANALYSIS_POLICY_CAMERA_MOBILE
        flow_options {
          fast_estimation_min_block_size: 100
          top_inlier_sets: 1
          frac_inlier_error_threshold: 3e-3
          downsample_mode: DOWNSAMPLE_TO_INPUT_SIZE
          # downsample_factor: 2.0
          verification_distance: 5.0
          verify_long_feature_acceleration: true
          verify_long_feature_trigger_ratio: 0.1
          tracking_options {
            max_features: 2000
            min_feature_distance: 4
            reuse_features_max_frame_distance: 3
            reuse_features_min_survived_frac: 0.9
            adaptive_extraction_levels: 2
            min_eig_val_settings {
              adaptive_lowest_quality_level: 2e-4
            }
            klt_tracker_implementation: KLT_OPENCV
          }
        }
        motion_options {
          label_empty_frames_as_valid: false
        }
      }
    }
  }
}

# Reads optical flow fields defined in
# mediapipe/framework/formats/motion/optical_flow_field.h,
# returns a VideoFrame with 2 channels (v_x and v_y), each channel is quantized
# to 0-255.
node: {
  calculator: "FlowPackagerCalculator"
  input_stream: "FLOW:region_flow"
  input_stream: "CAMERA:camera_motion"
  output_stream: "TRACKING:tracking_data"

  node_options: {
    [type.googleapis.com/mediapipe.FlowPackagerCalculatorOptions]: {
      flow_packager_options: {
        binary_tracking_data_support: false
      }
    }
  }
}

# Tracks box positions over time.
node: {
  calculator: "BoxTrackerCalculator"
  input_stream: "TRACKING:tracking_data"
  input_stream: "TRACK_TIME:input_frame"
  input_stream: "START_POS:start_pos"
  input_stream: "CANCEL_OBJECT_ID:cancel_object_id"
  input_stream_info: {
    tag_index: "CANCEL_OBJECT_ID"
    back_edge: true
  }
  output_stream: "BOXES:boxes"

  input_stream_handler {
    input_stream_handler: "SyncSetInputStreamHandler"
    options {
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
        sync_set {
          tag_index: "TRACKING"
          tag_index: "TRACK_TIME"
        }
        sync_set {
          tag_index: "START_POS"
        }
        sync_set {
          tag_index: "CANCEL_OBJECT_ID"
        }
      }
    }
  }

  node_options: {
    [type.googleapis.com/mediapipe.BoxTrackerCalculatorOptions]: {
      tracker_options: {
        track_step_options {
          track_object_and_camera: true
          tracking_degrees: TRACKING_DEGREE_TRANSLATION
          inlier_spring_force: 0.0
          static_motion_temporal_ratio: 3e-2
        }
      }
      visualize_tracking_data: false
      streaming_track_data_cache_size: 100
    }
  }
}


# Managers new detected objects and objects that are being tracked.
# It associates the duplicated detections and updates the locations of
# detections from tracking.
node: {
  calculator: "TrackedDetectionManagerCalculator"
  input_stream: "DETECTIONS:detections_with_id"
  input_stream: "TRACKING_BOXES:boxes"
  output_stream: "DETECTIONS:output_tracked"
  output_stream: "CANCEL_OBJECT_ID:cancel_object_id"

  options: {
    [mediapipe.TrackedDetectionManagerCalculatorOptions.ext]: {
      tracked_detection_manager_options {
        is_same_detection_min_overlap_ratio: 0.15
      }
    }
  }

  input_stream_handler {
    input_stream_handler: "SyncSetInputStreamHandler"
    options {
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
        sync_set {
          tag_index: "TRACKING_BOXES"
        }
        sync_set {
          tag_index: "DETECTIONS"
        }
      }
    }
  }
}
"""


class MediaPipeObjectTracker(BaseObjectDetectionInference):
  """MediaPipe-based tracking."""

  def __init__(self, object_detection_engine, config):
    del config  # Not used, yet.
    self._mediapipe_tracker = mediapipe_tracker.MediaPipeTracker(
        mediapipe_graph)
    self._object_detection_engine = object_detection_engine

  def input_size(self):
    return self._object_detection_engine.input_size()

  def run(self, timestamp, frame, annotations):
    np_frame = np.array(frame)

    detection_annotations = []
    if self._object_detection_engine.run(timestamp, np_frame,
                                         detection_annotations):
      converted_detections = []
      # Converts to MediaPipe Detection proto.
      for idx, annotation in enumerate(detection_annotations):
        detection = mediapipe_tracker.MediaPipeDetection()
        detection.timestamp_usec = timestamp
        detection.label = [annotation.class_name]
        detection.score = [annotation.confidence_score]
        detection.detection_id = idx
        location_data = mediapipe_tracker.LocationData()
        relative_bounding_box = mediapipe_tracker.RelativeBoundingBox()
        relative_bounding_box.xmin = annotation.bbox.left
        relative_bounding_box.ymin = annotation.bbox.top
        relative_bounding_box.width = annotation.bbox.right - annotation.bbox.left
        relative_bounding_box.height = annotation.bbox.bottom - annotation.bbox.top
        location_data.relative_bounding_box = relative_bounding_box
        detection.location_data = location_data
        converted_detections.append(detection)

      # Inputs annotations into mediapipe tracker.
      tracked_annotations = self._mediapipe_tracker.process(
          timestamp, converted_detections, np_frame)

      # Converts back to AutoML Video Edge detection structs.
      for tracked_annotation in tracked_annotations:
        highest_idx = tracked_annotation.score.index(
            max(tracked_annotation.score))
        output_annotation = ObjectTrackingAnnotation(
            timestamp=timestamp,
            track_id=tracked_annotation.detection_id,
            class_id=1 if tracked_annotation.label_id else -1,
            class_name=tracked_annotation.label[highest_idx]
            if tracked_annotation.label else '',
            confidence_score=tracked_annotation.score[highest_idx],
            bbox=NormalizedBoundingBox(
                left=tracked_annotation.location_data.relative_bounding_box
                .xmin,
                top=tracked_annotation.location_data.relative_bounding_box.ymin,
                right=tracked_annotation.location_data.relative_bounding_box
                .xmin +
                tracked_annotation.location_data.relative_bounding_box.width,
                bottom=tracked_annotation.location_data.relative_bounding_box
                .ymin +
                tracked_annotation.location_data.relative_bounding_box.height))
        annotations.append(output_annotation)
      return True
    else:
      return False
