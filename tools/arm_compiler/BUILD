licenses(["permissive"])

package(default_visibility = ["//visibility:public"])

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "arm64-v8a|clang": ":cc-compiler-local",
        "armeabi-v7a|clang": "cc-compiler-local",
        "k8|clang": ":cc-compiler-local",
        "arm64-v8a": ":cc-compiler-local",
        "armeabi-v7a": "cc-compiler-local",
        "k8": ":cc-compiler-local",
    },
)

filegroup(
    name = "empty",
    srcs = [],
)

cc_toolchain(
    name = "cc-compiler-local",
    all_files = ":empty",
    compiler_files = ":empty",
    cpu = "local",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 0,
)
