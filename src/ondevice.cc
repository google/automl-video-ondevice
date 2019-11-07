#include "third_party/automl_video_ondevice/src/ondevice.h"

#include <memory>

#include "mobile_lstd_tflite_client.h  // @lstm_object_detection"
#include "protos/box_encodings.pb.h  // @lstm_object_detection"
#include "protos/detections.pb.h  // @lstm_object_detection"
#include "protos/labelmap.pb.h  // @lstm_object_detection"
#include "protos/mobile_ssd_client_options.pb.h  // @lstm_object_detection"
#include "utils/file_utils.h  // @lstm_object_detection"

namespace automlvideo {
namespace ondevice {

namespace {
using ::google::protobuf::TextFormat;
using ::lstm_object_detection::tflite::MobileLSTDTfLiteClient;
using ::lstm_object_detection::tflite::protos::ClientOptions;
using ::lstm_object_detection::tflite::protos::DetectionResults;
using ::lstm_object_detection::tflite::protos::StringIntLabelMapProto;
}  // namespace

class TFLiteModelObjectTrackingInference : public ObjectTrackingInference {
 public:
  TFLiteModelObjectTrackingInference(const std::string &model_file,
                                     const std::string &label_map_file,
                                     const ObjectTrackingConfig &config) {
    auto options = MobileLSTDTfLiteClient::CreateDefaultOptions();
    options.set_quantize(true);
    options.set_score_threshold(config.score_threshold);
    options.mutable_external_files()->set_model_file_name(model_file);
    LOG(INFO) << "Loaded model.";
    LoadLabelMap(label_map_file, &options);
    LOG(INFO) << "Loaded label map.";
    detector_ = MobileLSTDTfLiteClient::Create(options);
    LOG(INFO) << "Client initialized.";
  }

  virtual bool run(const int64_t timestamp,
                   const std::vector<unsigned char> &frame,
                   std::vector<const ObjectTrackingAnnotation> *detections) {
    DetectionResults internal_detections;
    CHECK(detector_->Detect(frame.data(), &internal_detections));
    for (int i = 0; i < internal_detections.detection_size(); i++) {
      auto &internal_detection = internal_detections.detection(i);
      for (int j = 0; j < internal_detection.class_index_size(); ++j) {
        ObjectTrackingAnnotation detection;
        detection.timestamp = timestamp;
        detection.class_id = internal_detection.class_index(j);
        detection.class_name =
            detector_->GetLabelName(internal_detection.class_index(j));
        detection.confidence_score = internal_detection.score(j);
        detection.bbox.left = internal_detection.box().xmin(0);
        detection.bbox.top = internal_detection.box().ymin(0);
        detection.bbox.right = internal_detection.box().xmax(0);
        detection.bbox.bottom = internal_detection.box().ymax(0);
        detections->push_back(detection);
      }
    }
    return true;
  }

  virtual Size getInputSize() {
    return Size{detector_->GetInputWidth(), detector_->GetInputHeight()};
  }

 private:
  // Loads in the label map.
  //
  // Label map loading is a bit complicated due to the mobile client accepting
  // only the binary protobuffer whereas the ondevice library supports text
  // protobuffer. The text proto is deserialized then reserialized.
  bool LoadLabelMap(const std::string &label_map_file, ClientOptions *options) {
    StringIntLabelMapProto labelmap;
    const std::string proto_bytes =
        ::lstm_object_detection::tflite::ReadFileToString(label_map_file);
    // Parses text protobuffer.
    if (!TextFormat::ParseFromString(proto_bytes, &labelmap)) {
      return false;
    }
    // Serializes to binary std::string.
    auto labelmap_bytes = labelmap.SerializeAsString();
    options->mutable_external_files()->set_label_map_file_content(
        labelmap_bytes);
    return true;
  }

  std::unique_ptr<MobileLSTDTfLiteClient> detector_;
};

std::unique_ptr<ObjectTrackingInference> ObjectTrackingInference::TFLiteModel(
    const std::string &model_file, const std::string &label_map_file,
    const ObjectTrackingConfig &config) {
  return std::unique_ptr<ObjectTrackingInference>(
      new TFLiteModelObjectTrackingInference(model_file, label_map_file,
                                             config));
}

}  // namespace ondevice
}  // namespace automlvideo
