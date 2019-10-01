workspace(name = "automl_video_ondevice")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

http_archive(
    name = "bazel_skylib",
    sha256 = "bbccf674aa441c266df9894182d80de104cabd19be98be002f6d478aaa31574d",
    strip_prefix = "bazel-skylib-2169ae1c374aab4a09aa90e65efe1a3aad4e279b",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/2169ae1c374aab4a09aa90e65efe1a3aad4e279b.tar.gz"],
)

load("@bazel_skylib//lib:versions.bzl", "versions")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")
closure_repositories()

# ABSL cpp library.
http_archive(
    name = "com_google_absl",
    sha256 = "d10f684f170eb36f3ce752d2819a0be8cc703b429247d7d662ba5b4b48dd7f65",
    strip_prefix = "abseil-cpp-3088e76c597e068479e82508b1770a7ad0c806b6",
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/3088e76c597e068479e82508b1770a7ad0c806b6.tar.gz",
    ],
)

# GoogleTest/GoogleMock framework. Used by most unit-tests.
http_archive(
    name = "com_google_googletest",
    strip_prefix = "googletest-master",
    urls = ["https://github.com/google/googletest/archive/master.zip"],
)

# gflags needed by glog
http_archive(
    name = "com_github_gflags_gflags",
    strip_prefix = "gflags-2.2.2",
    urls = [
        "https://mirror.bazel.build/github.com/gflags/gflags/archive/v2.2.2.tar.gz",
        "https://github.com/gflags/gflags/archive/v2.2.2.tar.gz",
    ],
)

# glog
http_archive(
    name = "com_google_glog",
    # For security purpose, can use `sha256sum` on linux to calculate.
    sha256 = "835888ec47ee8065b3098f3ec4373717d641954970f009833ed6d466c397409a",
    strip_prefix = "glog-41f4bf9cbc3e8995d628b459f6a239df43c2b84a",
    urls = [
        "https://github.com/google/glog/archive/41f4bf9cbc3e8995d628b459f6a239df43c2b84a.tar.gz",
    ],
)

http_archive(
    name = "zlib",
    build_file = "@com_google_protobuf//:third_party/zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = ["https://zlib.net/zlib-1.2.11.tar.gz"],
)

http_archive(
    name = "gemmlowp",
    sha256 = "6678b484d929f2d0d3229d8ac4e3b815a950c86bb9f17851471d143f6d4f7834",
    strip_prefix = "gemmlowp-12fed0cd7cfcd9e169bf1925bc3a7a58725fdcc3",
    urls = [
        "http://mirror.tensorflow.org/github.com/google/gemmlowp/archive/12fed0cd7cfcd9e169bf1925bc3a7a58725fdcc3.zip",
        "https://github.com/google/gemmlowp/archive/12fed0cd7cfcd9e169bf1925bc3a7a58725fdcc3.zip",
    ],
)

#-----------------------------------------------------------------------------
# proto
#-----------------------------------------------------------------------------
# proto_library, cc_proto_library and java_proto_library rules implicitly depend
# on @com_google_protobuf//:proto, @com_google_protobuf//:cc_toolchain and
# @com_google_protobuf//:java_toolchain, respectively.
# This statement defines the @com_google_protobuf repo.
http_archive(
    name = "com_google_protobuf",
    sha256 = "1e622ce4b84b88b6d2cdf1db38d1a634fe2392d74f0b7b74ff98f3a51838ee53",
    strip_prefix = "protobuf-3.8.0",
    urls = ["https://github.com/google/protobuf/archive/v3.8.0.zip"],
)

# java_lite_proto_library rules implicitly depend on
# @com_google_protobuf_javalite//:javalite_toolchain, which is the JavaLite proto
# runtime (base classes and common utilities).
http_archive(
    name = "com_google_protobuf_javalite",
    sha256 = "79d102c61e2a479a0b7e5fc167bcfaa4832a0c6aad4a75fa7da0480564931bcc",
    strip_prefix = "protobuf-384989534b2246d413dbcd750744faab2607b516",
    urls = ["https://github.com/google/protobuf/archive/384989534b2246d413dbcd750744faab2607b516.zip"],
)

# Needed by TensorFlow
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "e0a111000aeed2051f29fcc7a3f83be3ad8c6c93c186e64beb1ad313f0c7f9f9",
    strip_prefix = "rules_closure-cf1e44edb908e9616030cc83d085989b8e6cd6df",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",  # 2019-04-04
    ],
)

# LSTM TFlite Inference
local_repository(
    name = "lstm_object_detection",
    path = "third_party/models/research/lstm_object_detection/tflite",
)

# Tensorflow
local_repository(
    name = "org_tensorflow",
    path = "third_party/tensorflow",
)
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(tf_repo_name = "org_tensorflow")

# EdgeTPU
local_repository(
    name = "libedgetpu",
    path = "third_party/edgetpu",
)

# shflags
http_archive(
    name = "shFlags",
    build_file = "//third_party:shFlags.BUILD",
    sha256 = "2f5cae06465d5eb98c5cf820e948b14ba994e493b17db579b2a57ef0ea2101a7",
    strip_prefix = "shflags-874a56a8ef6039ca8d77f8a883a239b62b4635ba",
    urls = ["https://github.com/kward/shflags/archive/874a56a8ef6039ca8d77f8a883a239b62b4635ba.zip"],
)

new_local_repository(
    name = "python_linux",
    path = "/",
    build_file = "BUILD.python"
)

local_repository(
    name = "tools",
    path = "third_party/edgetpu/tools",
)
load("@tools//:configure.bzl", "cc_crosstool")
cc_crosstool(name = "crosstool")
