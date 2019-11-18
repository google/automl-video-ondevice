package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "ondevice",
    srcs = ["libondevice.so"],
    hdrs = ["ondevice.h"],
    linkopts = ["-Wl,-R."],
)

cc_binary(
    name = "libondevice.so",
    srcs = [
        "ondevice.cc",
        "ondevice.h",
    ],
    linkshared = True,
    linkstatic = 1,
    deps = [
        "@lstm_object_detection//:mobile_lstd_tflite_client",
        "@lstm_object_detection//protos:box_encodings_cc_proto",
        "@lstm_object_detection//protos:detections_cc_proto",
        "@lstm_object_detection//protos:labelmap_cc_proto",
        "@lstm_object_detection//protos:mobile_ssd_client_options_cc_proto",
        "@lstm_object_detection//utils:file_utils",
    ],
)
