package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "ondevice",
    srcs = ["ondevice.cc"],
    hdrs = ["ondevice.h"],
    deps = [
        "@lstm_object_detection//:mobile_lstd_tflite_client",
        "@lstm_object_detection//protos:box_encodings_cc_proto",
        "@lstm_object_detection//protos:detections_cc_proto",
        "@lstm_object_detection//protos:labelmap_cc_proto",
        "@lstm_object_detection//protos:mobile_ssd_client_options_cc_proto",
        "@lstm_object_detection//utils:file_utils",
    ],
)
