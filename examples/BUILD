package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_binary(
    name = "ondevice_demo",
    srcs = ["ondevice_demo.cc"],
    deps = [
        ":inference_utils",
        "//third_party/automl_video_ondevice/src:ondevice",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_glog//:glog",
    ],
)

cc_library(
    name = "inference_utils",
    srcs = ["inference_utils.cc"],
    hdrs = ["inference_utils.h"],
    deps = [
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)
