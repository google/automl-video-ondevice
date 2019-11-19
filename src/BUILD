load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

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

cc_library(
    name = "ondevice_static",
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

cc_binary(
    name = "libondevice.so",
    linkshared = True,
    linkstatic = 1,
    deps = ["ondevice_static"],
)

pybind_extension(
    name = "automl_ondevice",
    srcs = ["ondevice_python_wrapper.cc"],
    copts = ["-Iexternal/python_37"],
    module_name = "automl_ondevice",
    srcs_version = "PY3ONLY",
    deps = [
        ":ondevice_static",
        "@pybind11",
        "@python_37//:headers",
        "@python_linux//:python3-headers",
    ],
)

py_test(
    name = "ondevice_python_wrapper_test",
    srcs = ["ondevice_python_wrapper_test.py"],
    data = [
        "//data",
    ],
    python_version = "PY3",
    deps = [
        ":automl_ondevice",
        "@absl_py//absl/testing:absltest",
    ],
)
