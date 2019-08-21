# Makefile for cross-compiling edgetpu/cpp/examples.
#
# Note that cross-compilation happens inside docker using bazel, this makefile
# here is mainly for convenience.
ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
OUT_DIR := cpp_example_out
TAG_EXAMPLE := coral/example-cross-compile
UID ?= $(shell id -u)
GID ?= $(shell id -g)
COPY := install -d -m 755 -o $(UID) -g $(GID) $(OUT_DIR) && install -C -m 755 -o $(UID) -g $(GID)
BAZEL_FLAGS := -c opt \
               --verbose_failures \
               --sandbox_debug \
               --crosstool_top=//tools/arm_compiler:toolchain \
               --compiler=clang \
			   --define enable_edgetpu=true
.PHONY: all \
        docker-example-image \
        docker-example-shell \
        docker-example-compile \
        amd64 \
        arm64 \
        arm32 \
        clean
all:
	@echo "make docker-example-image   - Build docker image for cpp examples"
	@echo "make docker-example-shell   - Run shell to docker image for cpp examples"
	@echo "make docker-example-compile - Compile cpp examples for all platforms"
	@echo "make examples               - Compile cpp examples for amd64, arm64, arm32"
	@echo "make clean               - Remove generated files"
docker-example-image:
	docker build -t $(TAG_EXAMPLE) -f tools/Dockerfile.16.04 tools
docker-example-shell: docker-example-image
	docker run --rm -it -v $(ROOT_DIR):/edgetpu-ml-cpp $(TAG_EXAMPLE)
docker-example-compile: docker-example-image
	docker run --rm -t -v $(ROOT_DIR):/edgetpu-ml-cpp $(TAG_EXAMPLE) make -f example.makefile UID=$(UID) GID=$(GID) -C /edgetpu-ml-cpp ondevice-examples
ondevice-examples: ondevice-examples-amd64 ondevice-examples-arm64 ondevice-examples-arm32
ondevice-examples-amd64:
	bazel build $(BAZEL_FLAGS) --features=glibc_compat --cpu=k8 //examples:ondevice_demo
	$(COPY) bazel-out/k8-opt/bin/examples/ondevice_demo $(OUT_DIR)/ondevice_demo_amd64
ondevice-examples-arm64:
	bazel build $(BAZEL_FLAGS) --cpu=arm64-v8a //examples:ondevice_demo
	$(COPY) bazel-out/arm64-v8a-opt/bin/examples/ondevice_demo $(OUT_DIR)/ondevice_demo_arm64
ondevice-examples-arm32:
	bazel build $(BAZEL_FLAGS) --cpu=armeabi-v7a //examples:ondevice_demo
	$(COPY) bazel-out/armeabi-v7a-opt/bin/examples/ondevice_demo $(OUT_DIR)/ondevice_demo_arm32
clean:
	rm -rf $(OUT_DIR)