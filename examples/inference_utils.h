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

#ifndef EXAMPLES_INFERENCE_UTILS_H_
#define EXAMPLES_INFERENCE_UTILS_H_

#include <array>
#include <string>
#include <vector>

#include "tensorflow/lite/interpreter.h"

namespace automlvideo {
namespace ondevice {

// Defines dimension of an image, in height, width, depth order.
typedef std::array<int, 3> ImageDims;

// The box is represented by a vector with 4 coordinates: x1, y2, x2, y2.
// First point is left-top while second is right bottom.
typedef std::array<float, 4> Box;

// Returns total number of elements.
int ImageDimsToSize(const ImageDims& dims);

// Reads BMP image. Returns empty vector upon failure.
std::vector<uint8_t> ReadBmp(const std::string& input_bmp_name,
                             ImageDims* image_dims);

// Resizes BMP image.
void ResizeImage(const ImageDims& in_dims, const uint8_t* in,
                 const ImageDims& out_dims, uint8_t* out);

// Gets input from images and resizes to `target_dims`. Returns empty vector
// upon failure.
std::vector<uint8_t> GetInputFromImage(const std::string& image_path,
                                       const ImageDims& target_dims);

// Finds all BMP files in a given directory.
std::vector<std::string> FindImages(const std::string& directory);

}  // namespace ondevice
}  // namespace automlvideo

#endif  // EXAMPLES_INFERENCE_UTILS_H_
