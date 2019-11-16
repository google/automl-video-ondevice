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

#include <fstream>
#include <iostream>

#include "absl/strings/str_format.h"
#include "examples/inference_utils.h"
#include "gflags/gflags.h"
#include "src/ondevice.h"

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
  ObjectTrackingConfig config;
  config.score_threshold = 0.2f;
  const auto inference = ObjectTrackingInference::TFLiteModel(
      FLAGS_model_file_path, FLAGS_label_map_file_path, config);

  // Retrieves all images in a given directory as a list.
  std::vector<std::string> image_files = FindImages(FLAGS_images_file_path);
  std::sort(image_files.begin(), image_files.end());

  // Gets the input size compatible with the inference graph.
  // This is used to resize the images to something the inferencer can use.
  Size input_size = inference->getInputSize();

  // Loops through all images in directory.
  for (const std::string &image_file : image_files) {
    LOG(INFO) << "Input image: " << image_file;

    // Reads image into a 256x256x3 byte array.
    std::vector<uint8_t> input =
        GetInputFromImage(image_file, {input_size.width, input_size.height, 3});

    // Opens up detections result file for writing.
    std::ofstream detections_file;
    detections_file.open(image_file + ".txt",
                         std::ofstream::out | std::ofstream::trunc);

    // Runs inference.
    std::vector<ObjectTrackingAnnotation> annotations;
    int64_t timestamp = 0;
    if (inference->run(timestamp++, input, &annotations)) {
      for (auto annotation : annotations) {
        std::string annotations_entry = ::absl::StrFormat(
            "%s: %f [%f, %f, %f, %f]\n", annotation.class_name,
            annotation.confidence_score, annotation.bbox.top,
            annotation.bbox.left, annotation.bbox.bottom,
            annotation.bbox.right);
        LOG(INFO) << annotations_entry;
        detections_file << annotations_entry;
      }
    } else {
      LOG(WARNING) << "Could not run inference on input image!";
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
