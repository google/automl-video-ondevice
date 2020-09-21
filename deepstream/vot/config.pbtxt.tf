name: "vot"
platform: "tensorflow_graphdef"
default_model_filename: "frozen_inference_graph.pb"

max_batch_size: 128
input [
  {
    name: "image_tensor"
    data_type: TYPE_UINT8
    format: FORMAT_NHWC
    dims: [ 256,256, 3 ]
  }
]
output [
  {
    name: "detection_boxes"
    data_type: TYPE_FP32
    dims: [ 100, 4]
    reshape { shape: [100, 4] }
  },
  {
    name: "detection_classes"
    data_type: TYPE_FP32
    dims: [ 100 ]
  },
  {
    name: "detection_scores"
    data_type: TYPE_FP32
    dims: [ 100]
  },
  {
    name: "num_detections"
    data_type: TYPE_FP32
    dims: [ 1 ]
    reshape { shape: [] }
  }
]

optimization { execution_accelerators {
  gpu_execution_accelerator : [ {
    name : "tensorrt"
    parameters { key: "precision_mode" value: "FP16" }}]
}}

