#ifndef AUTOML_VIDEO_ONDEVICE_SRC_ONDEVICE_H_
#define AUTOML_VIDEO_ONDEVICE_SRC_ONDEVICE_H_

#include <memory>
#include <string>
#include <vector>

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
  int64_t timestamp;  // User-defined integer, identifying which frame this
                      // annotation is associated to.

  int track_id = -1;  // If applicable, a unique ID of the object tracked
                      // throughout the entire inference run.
                      // (Optionally provided.)

  int class_id;  // Classification ID of object.

  std::string class_name;  // Human-readable classification name.

  float confidence_score;  // Confidence score (The higher the better).
                           // Range: [0.0, 1.0].

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

// ObjectTrackingInference is an abstract class defining a basic object
// detection + tracking interface.
//
// Several implementations of this class will be available, which will allow
// object tracking on many different devices and platforms.
//
// Example intialization for TFLite inference:
//   const auto inference = ObjectTrackingInference::TFLiteModel(
//     "model.pb", "labelmap.pbtxt", {/*score_threshold=*/0.5f});
//
// Preprocessing is not handled, so the image must be resized by the user.
// The necessary input image size can be retrieved using:
//    inference.getInputSize().
//
// Finally, to run inferencing and retrieve annotations:
//    std::vector<const ObjectTrackingAnnotation> annotations;
//    inference.run(100, frame, &annotations);
//
class ObjectTrackingInference {
 public:
  virtual ~ObjectTrackingInference() {}

  // A blocking function that runs inferencing on a single image frame.
  // Timestamp can be any integer, and will be passed back through the
  // annotations. Used to associate annotations with frames. Uniqueness is not
  // checked. For implementations using tracking, this must be an accurate
  // millisecond integer. Frame must be in RGB888 format.
  //
  // Returns true if inference is successful.
  virtual bool run(const int64_t timestamp,
                   const std::vector<unsigned char> &frame,
                   std::vector<ObjectTrackingAnnotation> *annotations) = 0;

  // If available, retrieves the input size accepted by the model.
  virtual Size getInputSize() = 0;

  // Loads in a TFLite model. Accepts a TFLite model path and label map path.
  // Only outputs detections, tracking data is not available.
  static std::unique_ptr<ObjectTrackingInference> TFLiteModel(
      const std::string &model_file, const std::string &label_map_file,
      const ObjectTrackingConfig &config = ObjectTrackingConfig());
};

}  // namespace ondevice
}  // namespace automlvideo

#endif  // AUTOML_VIDEO_ONDEVICE_SRC_ONDEVICE_H_
