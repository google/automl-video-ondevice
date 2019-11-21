#include <string>
#include <vector>

#include "include/pybind11/numpy.h"
#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"
#include "include/pybind11/stl.h"
#include "src/ondevice.h"

namespace py = pybind11;

using automlvideo::ondevice::NormalizedBoundingBox;
using automlvideo::ondevice::ObjectTrackingAnnotation;
using automlvideo::ondevice::ObjectTrackingConfig;
using automlvideo::ondevice::ObjectTrackingInference;
using automlvideo::ondevice::Size;

PYBIND11_MODULE(automl_ondevice, m) {
  // Create a temporary ObjectTrackingConfig to get the default values.
  constexpr ObjectTrackingConfig defaultObjectTrackingConfig_;
  py::class_<ObjectTrackingConfig>(m, "ObjectTrackingConfig")
      .def(py::init([](float score_threshold, int max_detections) {
             ObjectTrackingConfig config;
             config.score_threshold = score_threshold;
             config.max_detections = max_detections;
             return config;
           }),
           py::arg("score_threshold") =
               defaultObjectTrackingConfig_.score_threshold,
           py::arg("max_detections") =
               defaultObjectTrackingConfig_.max_detections)
      .def_readwrite("score_threshold", &ObjectTrackingConfig::score_threshold)
      .def_readwrite("max_detections", &ObjectTrackingConfig::max_detections);

  py::class_<ObjectTrackingInference>(m, "ObjectTrackingInference")
      .def_static("TFLiteModel", &ObjectTrackingInference::TFLiteModel,
                  py::arg("model_file"), py::arg("label_map_file"),
                  py::arg("config"))
      .def("getInputSize", &ObjectTrackingInference::getInputSize)
      .def("run",
           [](ObjectTrackingInference &self, const int64_t timestamp,
              const py::array_t<unsigned char> &frame, py::list &annotations) {
             const auto input_size = self.getInputSize();
             if (frame.size() != input_size.width * input_size.height * 3) {
               return false;
             }

             std::vector<unsigned char> cpp_frame;
             cpp_frame.reserve(frame.size());

             // TODO: Find a way to directly refer to the data rather than make
             // a copy.
             std::memcpy(cpp_frame.data(), frame.data(), frame.size());
             std::vector<ObjectTrackingAnnotation> cpp_annotations;
             if (!self.run(timestamp, cpp_frame, &cpp_annotations)) {
               return false;
             }

             // Fills output array with annotations.
             for (ObjectTrackingAnnotation annotation : cpp_annotations) {
               annotations.append(annotation);
             }
             return true;
           });

  py::class_<NormalizedBoundingBox>(m, "NormalizedBoundingBox")
      .def_readonly("left", &NormalizedBoundingBox::left)
      .def_readonly("top", &NormalizedBoundingBox::top)
      .def_readonly("right", &NormalizedBoundingBox::right)
      .def_readonly("bottom", &NormalizedBoundingBox::bottom)
      .def("__repr__", [](const NormalizedBoundingBox &a) {
        return "<automl_ondevice.NormalizedBoundingBox left: " +
               std::to_string(a.left) + ", top: " + std::to_string(a.top) +
               ", right: " + std::to_string(a.right) +
               ", bottom: " + std::to_string(a.bottom) + ">";
      });

  py::class_<ObjectTrackingAnnotation>(m, "ObjectTrackingAnnotation")
      .def_readonly("timestamp", &ObjectTrackingAnnotation::timestamp)
      .def_readonly("track_id", &ObjectTrackingAnnotation::track_id)
      .def_readonly("class_id", &ObjectTrackingAnnotation::class_id)
      .def_readonly("class_name", &ObjectTrackingAnnotation::class_name)
      .def_readonly("confidence_score",
                    &ObjectTrackingAnnotation::confidence_score)
      .def_readonly("bbox", &ObjectTrackingAnnotation::bbox);

  py::class_<Size>(m, "Size")
      .def_readonly("width", &Size::width)
      .def_readonly("height", &Size::height)
      .def("__repr__", [](const Size &a) {
        return "<automl_ondevice.Size width: " + std::to_string(a.width) +
               ", height: " + std::to_string(a.height) + ">";
      });
};
