// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
//
// Demo binary that shows simple prediction from a Google Cloud AutoML Video
// trained LSTD MobileNet V2 model in TfLite format.

#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <iostream>

#include "absl/strings/str_format.h"
#include "examples/inference_utils.h"
#include "gflags/gflags.h"
#include "mobile_lstd_tflite_client.h  // @lstm_object_detection"
#include "protos/box_encodings.pb.h  // @lstm_object_detection"
#include "protos/detections.pb.h  // @lstm_object_detection"
#include "protos/labelmap.pb.h  // @lstm_object_detection"
#include "utils/file_utils.h  // @lstm_object_detection"

DEFINE_string(images_file_path, "/tmp/car",
              "directory of images to run inference on");
DEFINE_string(model_file_path, "/tmp/model.tflite", "model file path");
DEFINE_string(label_map_file_path, "/tmp/label_map.pbtxt",
              "label map file path");

namespace automlvideo {
namespace ondevice {

namespace {
struct Result {
  int left;
  int right;
  int top;
  int bottom;
  std::string label;
};
}  // namespace

void DetectMain() {
  // Retrieves all images in a given directory as a list.
  std::vector<std::string> image_files = FindImages(FLAGS_images_file_path);
  std::sort(image_files.begin(), image_files.end());

  // Loads tflite model. Supports Edge and Non-Edge models.
  auto options = ::lstm_object_detection::tflite::MobileLSTDTfLiteClient::
      CreateDefaultOptions();
  options.set_quantize(true);
  options.set_score_threshold(-5.0);
  CHECK(!FLAGS_model_file_path.empty());
  options.mutable_external_files()->set_model_file_name(FLAGS_model_file_path);
  LOG(INFO) << "Loaded model.";

  // Loads label map.
  CHECK(!FLAGS_label_map_file_path.empty());
  ::lstm_object_detection::tflite::protos::StringIntLabelMapProto labelmap;
  const std::string proto_bytes =
      ::lstm_object_detection::tflite::ReadFileToString(
          FLAGS_label_map_file_path);
  CHECK(
      ::google::protobuf::TextFormat::ParseFromString(proto_bytes, &labelmap));
  auto labelmap_bytes = labelmap.SerializeAsString();
  options.mutable_external_files()->set_label_map_file_content(labelmap_bytes);
  LOG(INFO) << "Loaded label map.";

  auto detector = ::lstm_object_detection::tflite::MobileLSTDTfLiteClient::
      MobileLSTDTfLiteClient::Create(options);

  // Loops through all images in directory.
  for (const std::string &image_file : image_files) {
    LOG(INFO) << "Input image: " << image_file;

    // Reads image into a 256x256x3 byte array.
    std::vector<uint8_t> input = GetInputFromImage(
        image_file, {detector->GetInputWidth(), detector->GetInputHeight(), 3});

    // Runs inference.
    ::lstm_object_detection::tflite::protos::DetectionResults detections;
    CHECK(detector->Detect(input.data(), &detections));

    // Opens up detections result file for writing.
    std::ofstream detections_file;
    detections_file.open(image_file + ".txt",
                         std::ofstream::out | std::ofstream::trunc);
    for (int i = 0; i < detections.detection_size(); i++) {
      auto &det = detections.detection(i);
      for (int j = 0; j < det.class_index_size(); ++j) {
        std::string class_name = detector->GetLabelName(det.class_index(j));
        std::string detections_entry = ::absl::StrFormat(
            "%s: %f [%f, %f, %f, %f]\n", class_name, det.score(j),
            det.box().ymin(0), det.box().xmin(0), det.box().ymax(0),
            det.box().xmax(0));
        LOG(INFO) << detections_entry;
        detections_file << detections_entry;
      }
    }
    detections_file.close();
  }
}
}  // namespace ondevice
}  // namespace automlvideo

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::automlvideo::ondevice::DetectMain();
  return 0;
}
