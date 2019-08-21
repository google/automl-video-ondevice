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
#include "mobile_lstd_tflite_client.h"
#include "protos/box_encodings.pb.h"
#include "protos/detections.pb.h"
#include "protos/labelmap.pb.h"
#include "utils/file_utils.h"

DEFINE_string(images_file_path, "/tmp/car/",
              "directory of images to run inference on");
DEFINE_string(model_file_path, "/tmp/yongzhe_no_anchor_model.tflite",
              "model file path");
DEFINE_string(label_map_file_path, "/tmp/labelmap.pbtxt",
              "label map file path");

namespace automlvideo {
namespace ondevice {

struct Result {
  int left;
  int right;
  int top;
  int bottom;
  std::string label;
};

void DetectMain() {
  // List the files to process.
  LOG(INFO) << "image file pattern" << FLAGS_images_file_path;
  std::vector<std::string> image_files = FindImages(FLAGS_images_file_path);

  std::sort(image_files.begin(), image_files.end());
  std::map<std::string, std::vector<std::vector<Result>>> results;

  auto options = ::lstm_object_detection::tflite::MobileLSTDTfLiteClient::
      CreateDefaultOptions();
  options.set_quantize(true);
  options.set_score_threshold(-5.0);
  CHECK(!FLAGS_model_file_path.empty());
  options.mutable_external_files()->set_model_file_name(FLAGS_model_file_path);
  LOG(INFO) << "Loaded Client";
  CHECK(!FLAGS_label_map_file_path.empty());
  ::lstm_object_detection::tflite::protos::StringIntLabelMapProto labelmap;
  const std::string proto_bytes =
      ::lstm_object_detection::tflite::ReadFileToString(
          FLAGS_label_map_file_path);
  CHECK(
      ::google::protobuf::TextFormat::ParseFromString(proto_bytes, &labelmap));
  auto labelmap_bytes = labelmap.SerializeAsString();
  options.mutable_external_files()->set_label_map_file_content(labelmap_bytes);
  LOG(INFO) << "Label map";

  auto detector = ::lstm_object_detection::tflite::MobileLSTDTfLiteClient::
      MobileLSTDTfLiteClient::Create(options);

  ::lstm_object_detection::tflite::protos::DetectionResults detections;
  LOG(INFO) << "image file size: " << image_files.size();
  for (const std::string &image_file : image_files) {
    LOG(INFO) << "Input image: " << image_file;

    // Read image
    std::vector<uint8_t> input = GetInputFromImage(
        image_file, {detector->GetInputWidth(), detector->GetInputHeight(), 3});
    CHECK(detector->Detect(input.data(), &detections));
    std::vector<Result> image_results;
    std::ofstream detections_file;
    detections_file.open(image_file + ".txt", std::ios::trunc);
    for (int i = 0; i < detections.detection_size(); i++) {
      auto &det = detections.detection(i);
      LOG(INFO) << ::absl::StrFormat(
          "Detection: box [% .2f % .2f % .2f % .2f] ", det.box().ymin(0),
          det.box().xmin(0), det.box().ymax(0), det.box().xmax(0));
      for (int j = 0; j < det.class_index_size(); ++j) {
        std::string class_name = detector->GetLabelName(det.class_index(j));
        LOG(INFO) << ::absl::StrFormat("  %i: object class [%s], score [%0.2f]",
                                       j + 1, class_name, det.score(j));
        detections_file << ::absl::StrFormat(
            "%s: %f [%f, %f, %f, %f]\n", class_name, det.score(j),
            det.box().ymin(0), det.box().xmin(0), det.box().ymax(0),
            det.box().xmax(0));
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

// Needed when compiling with EdgeTPU chef release. Can be removed when
// upgraded to diploria release.
extern "C" int __cxa_thread_atexit(void (*func)(), void *obj,
                                   void *dso_symbol) {
  int __cxa_thread_atexit_impl(void (*)(), void *, void *);
  return __cxa_thread_atexit_impl(func, obj, dso_symbol);
}