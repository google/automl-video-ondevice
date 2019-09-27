# Makefile for cross-compiling automl video ondevice examples.
#
# Note that cross-compilation happens inside docker using bazel, this makefile
# here is mainly for convenience.
CUR_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
OUT_DIR := bin
TAG_EXAMPLE := automl-video-ondevice/examples
UID ?= $(shell id -u)
GID ?= $(shell id -g)
COPY := install -d -m 755 -o $(UID) -g $(GID) $(OUT_DIR) && install -C -m 755 -o $(UID) -g $(GID)
BAZEL_FLAGS := --crosstool_top=@crosstool//:toolchains \
				--compilation_mode=opt \
				--copt=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
				--verbose_failures \
				--sandbox_debug \
				--compiler=gcc \
				--subcommands \
				--define enable_edgetpu=true \
				--linkopt=-l:libedgetpu.so.1.0 \
				--linkopt=-Wl,--strip-all \
				--copt=-ffp-contract=off

.PHONY: all \
				docker-example-image \
				docker-example-shell \
				docker-example-compile \
				ondevice-examples \
				clean
all:
	@echo "make docker-example-image   - Build docker image for cpp examples"
	@echo "make docker-example-shell   - Run shell to docker image for cpp examples"
	@echo "make docker-example-compile - Compile cpp examples for all platforms"
	@echo "make ondevice-examples      - Compile cpp examples for k8, armv7a, aarch64"
	@echo "make clean                  - Remove generated files"

docker-example-image:
	docker build -t $(TAG_EXAMPLE) --build-arg VERSION="18.04" -f third_party/edgetpu/docker/Dockerfile.ubuntu third_party/edgetpu/docker
docker-example-shell: docker-example-image
	docker run --rm -it -v $(ROOT_DIR):/automl-video-ondevice $(TAG_EXAMPLE)
docker-example-compile: docker-example-image
	docker run --rm -t -v $(ROOT_DIR):/automl-video-ondevice $(TAG_EXAMPLE) make UID=$(UID) GID=$(GID) -C /automl-video-ondevice ondevice-examples

ondevice-examples: ondevice-examples-k8 ondevice-examples-armv7a ondevice-examples-aarch64

ondevice-examples-k8:
	bazel build $(BAZEL_FLAGS) --cpu=k8 \
		--linkopt=-L$(CUR_DIR)/third_party/edgetpu/libedgetpu/direct/k8 \
		//examples:ondevice_demo
	$(COPY) bazel-out/k8-opt/bin/examples/ondevice_demo $(OUT_DIR)/ondevice_demo_k8
ondevice-examples-armv7a:
	bazel build $(BAZEL_FLAGS) --cpu=armv7a \
		--linkopt=-L$(CUR_DIR)/third_party/edgetpu/libedgetpu/direct/armv7a \
		//examples:ondevice_demo
	$(COPY) bazel-out/armv7a-opt/bin/examples/ondevice_demo $(OUT_DIR)/ondevice_demo_armv7a
ondevice-examples-aarch64:
	bazel build $(BAZEL_FLAGS) --cpu=aarch64 \
		--linkopt=-L$(CUR_DIR)/third_party/edgetpu/libedgetpu/direct/aarch64 \
		//examples:ondevice_demo
	$(COPY) bazel-out/aarch64-opt/bin/examples/ondevice_demo $(OUT_DIR)/ondevice_demo_aarch64

clean:
	rm -rf $(OUT_DIR)