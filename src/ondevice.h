#ifndef AUTOML_VIDEO_ONDEVICE_SRC_ONDEVICE_H_
#define AUTOML_VIDEO_ONDEVICE_SRC_ONDEVICE_H_

#include <memory>
#include <string>

namespace automlvideo {
namespace ondevice {

// Normalized bounding box.
// The normalized vertex coordinates are relative to the original image.
// Range: [0.0, 1.0].
struct NormalizedBoundingBox {
  float left;
  float top;
  float right;
  float bottom;
};

// Annotations containing detection boxes, classes, and additional information.
struct ObjectTrackingAnnotation {
  int track_id = -1;       // If applicable, a unique ID of the object tracked
                           // throughout the entire inference run.
                           // (Optionally provided.)
  int class_id;            // Classification ID of object.
  std::string class_name;  // Human-readable classification ID.
  float confidence_score;
  NormalizedBoundingBox bbox;
};

// Object tracking and inferencing configurations.
struct ObjectTrackingConfig {
  float score_threshold = 0.0f;  // Minimum score threshold. Range: [0.0, 1.0]
  int max_detections = 100;      // Minimum amount of detections to return.
};

struct Size {
  int width;
  int height;
};

class ObjectTrackingInference {
 public:
  virtual ~ObjectTrackingInference() {}

  // Runs inferencing on a single image frame.
  // Frame must be in RGB888 format.
  virtual bool run(
      const std::vector<unsigned char> &frame,
      std::vector<const ObjectTrackingAnnotation> *annotations) = 0;

  // If available, retrieves the input size accepted by the model.
  virtual Size getInputSize() = 0;

  // Loads in a TFLite model. Accepts a TFLite model path and label map path.
  static std::unique_ptr<ObjectTrackingInference> TFLiteModel(
      const std::string &model_file, const std::string &label_map_file,
      const ObjectTrackingConfig &config = ObjectTrackingConfig());
};

}  // namespace ondevice
}  // namespace automlvideo

#endif  // AUTOML_VIDEO_ONDEVICE_SRC_ONDEVICE_H_
