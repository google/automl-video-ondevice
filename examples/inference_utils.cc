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

#include "examples/inference_utils.h"

#include <dirent.h>
#include <glog/logging.h>
#include <sys/types.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#include "absl/strings/match.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"

namespace automlvideo {
namespace ondevice {
namespace {

using tflite::ops::builtin::BuiltinOpResolver;

std::vector<uint8_t> _DecodeBmp(const uint8_t* input, int row_size, int width,
                                int height, int channels, bool top_down) {
  std::vector<uint8_t> output(height * width * channels);
  for (int i = 0; i < height; i++) {
    int src_pos;
    int dst_pos;
    for (int j = 0; j < width; j++) {
      if (!top_down) {
        src_pos = ((height - 1 - i) * row_size) + j * channels;
      } else {
        src_pos = i * row_size + j * channels;
      }
      dst_pos = (i * width + j) * channels;
      switch (channels) {
        case 1:
          output[dst_pos] = input[src_pos];
          break;
        case 3:
          // BGR -> RGB
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          break;
        case 4:
          // BGRA -> RGBA
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          output[dst_pos + 3] = input[src_pos + 3];
          break;
        default:
          LOG(FATAL) << "Unexpected number of channels: " << channels;
          break;
      }
    }
  }
  return output;
}
}  // namespace

void ResizeImage(const ImageDims& in_dims, const uint8_t* in,
                 const ImageDims& out_dims, uint8_t* out) {
  const int image_height = in_dims[0];
  const int image_width = in_dims[1];
  const int image_channels = in_dims[2];
  const int wanted_height = out_dims[0];
  const int wanted_width = out_dims[1];
  const int wanted_channels = out_dims[2];
  int number_of_pixels = image_height * image_width * image_channels;
  std::unique_ptr<tflite::Interpreter> interpreter(new tflite::Interpreter);
  int base_index = 0;
  // two inputs: input and new_sizes
  interpreter->AddTensors(2, &base_index);
  // one output
  interpreter->AddTensors(1, &base_index);
  // set input and output tensors
  interpreter->SetInputs({0, 1});
  interpreter->SetOutputs({2});
  // set parameters of tensors
  TfLiteQuantizationParams quant;
  interpreter->SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "input",
      {1, image_height, image_width, image_channels}, quant);
  interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2},
                                            quant);
  interpreter->SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "output",
      {1, wanted_height, wanted_width, wanted_channels}, quant);
  BuiltinOpResolver resolver;
  const TfLiteRegistration* resize_op =
      resolver.FindOp(tflite::BuiltinOperator_RESIZE_BILINEAR, 1);
  auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(
      malloc(sizeof(TfLiteResizeBilinearParams)));
  params->align_corners = false;
  interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op,
                                     nullptr);
  interpreter->AllocateTensors();
  // fill input image
  // in[] are integers, cannot do memcpy() directly
  auto input = interpreter->typed_tensor<float>(0);
  for (int i = 0; i < number_of_pixels; i++) {
    input[i] = in[i];
  }
  // fill new_sizes
  interpreter->typed_tensor<int>(1)[0] = wanted_height;
  interpreter->typed_tensor<int>(1)[1] = wanted_width;
  interpreter->Invoke();
  auto output = interpreter->typed_tensor<float>(2);
  auto output_number_of_pixels =
      wanted_height * wanted_height * wanted_channels;
  for (int i = 0; i < output_number_of_pixels; i++) {
    out[i] = static_cast<uint8_t>(output[i]);
  }
}

int ImageDimsToSize(const ImageDims& dims) {
  int size = 1;
  for (const auto& dim : dims) {
    size *= dim;
  }
  return size;
}

std::vector<uint8_t> ReadBmp(const std::string& input_bmp_name,
                             ImageDims* image_dims) {
  int* height = image_dims->data();
  int* width = image_dims->data() + 1;
  int* channels = image_dims->data() + 2;
  int begin, end;
  std::ifstream file(input_bmp_name, std::ios::in | std::ios::binary);
  if (!file) return {};

  begin = file.tellg();
  file.seekg(0, std::ios::end);
  end = file.tellg();
  size_t len = end - begin;
  VLOG(1) << "len: " << len << "\n";
  std::vector<uint8_t> img_bytes(len);
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(img_bytes.data()), len);
  const int32_t header_size =
      *(reinterpret_cast<const int32_t*>(img_bytes.data() + 10));
  *width = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 18));
  *height = *(reinterpret_cast<const int32_t*>(img_bytes.data() + 22));
  const int32_t bpp =
      *(reinterpret_cast<const int32_t*>(img_bytes.data() + 28));
  *channels = bpp / 8;
  if (*width < 0 || *height < 0 || *channels != 3) return {};
  VLOG(1) << "width, height, channels: " << *width << ", " << *height << ", "
          << *channels << "\n";

  // there may be padding bytes when the width is not a multiple of 4 bytes
  // 8 * channels == bits per pixel
  const int row_size = (8 * *channels * *width + 31) / 32 * 4;
  // if height is negative, data layout is top down
  // otherwise, it's bottom up
  bool top_down = (*height < 0);
  // Decode image, allocating tensor once the image size is known
  const uint8_t* bmp_pixels = &img_bytes[header_size];
  return _DecodeBmp(bmp_pixels, row_size, *width, abs(*height), *channels,
                    top_down);
}

std::vector<uint8_t> GetInputFromImage(const std::string& image_path,
                                       const ImageDims& target_dims) {
  std::vector<uint8_t> result;
  if (!::absl::EndsWithIgnoreCase(image_path, ".bmp")) {
    LOG(FATAL) << "Unsupported image type: " << image_path;
    return result;
  }
  result.resize(ImageDimsToSize(target_dims));
  ImageDims image_dims;
  std::vector<uint8_t> in = ReadBmp(image_path, &image_dims);
  if (in.empty()) return {};
  ResizeImage(image_dims, in.data(), target_dims, result.data());
  return result;
}

std::vector<std::string> FindImages(const std::string& directory) {
  std::string directory_with_ending_slash;
  if (::absl::EndsWith(directory, "/")) {
    directory_with_ending_slash = directory;
  } else {
    directory_with_ending_slash = directory + "/";
  }

  std::vector<std::string> paths;
  DIR* d;
  struct dirent* dir;
  d = opendir(directory_with_ending_slash.c_str());
  if (d) {
    while ((dir = readdir(d)) != NULL) {
      if (::absl::EndsWithIgnoreCase(dir->d_name, ".bmp")) {
        paths.push_back(directory_with_ending_slash + dir->d_name);
      }
    }
    closedir(d);
  }
  return paths;
}

}  // namespace ondevice
}  // namespace automlvideo
