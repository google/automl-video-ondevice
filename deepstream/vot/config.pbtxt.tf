name: "vot"
platform: "tensorflow_graphdef"
default_model_filename: "frozen_inference_graph.pb"

max_batch_size: 128
input [
  {
    name: "image_tensor"
    data_type: TYPE_UINT8
    format: FORMAT_NHWC
    dims: [ 300, 300, 3 ]
  }
]
output [
  {
    name: "detection_boxes"
    data_type: TYPE_FP32
    dims: [ 40, 4]
    reshape { shape: [40, 4] }
  },
  {
    name: "detection_classes"
    data_type: TYPE_FP32
    dims: [ 40 ]
  },
  {
    name: "detection_scores"
    data_type: TYPE_FP32
    dims: [ 40]
  },
  {
    name: "num_detections"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape { shape: [] }
  }
]

