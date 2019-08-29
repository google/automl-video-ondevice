package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "lib",
    srcs = ["libedgetpu_direct_all.so"],
)

cc_binary(
    name = "ondevice_demo",
    srcs = ["ondevice_demo.cc"],
    deps = [
        ":inference_utils",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_glog//:glog",
        "@lstm_object_detection//:mobile_lstd_tflite_client",
        "@lstm_object_detection//protos:box_encodings_cc_proto",
        "@lstm_object_detection//protos:detections_cc_proto",
        "@lstm_object_detection//protos:labelmap_cc_proto",
        "@lstm_object_detection//utils:file_utils",
        # "@libedgetpu//libedgetpu:lib",
        # ":lib",
    ],
)

cc_library(
    name = "inference_utils",
    srcs = ["inference_utils.cc"],
    hdrs = ["inference_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)
