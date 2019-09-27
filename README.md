# AutoML Video On-Device Examples

The example code that shows how to load the Google Cloud AutoML Video Object
Tracking On-Device models and conduct inference on a sequence of images from a
video clip.

The targeted devices are CPU and Edge TPU.

## Maintainers
* yongzhe@google.com
* henryquoctran@google.com

---

## Dependencies
Docker (>= 18.09.3)

Docker is used to simplify the development process and avoid any
environment and setup issues.

## Build Examples
To build the example in one-shot:

```
make docker-example-compile
```

## Developing
The above command will instantly remove all build artifacts and caches
upon completion, and can be slow for active development.

For development, launch a docker shell instead:

```
make docker-example-shell
cd automl-video-ondevice
```

Then to build:

```
make ondevice-examples-${CPU}
```

Where CPU can be:

* k8
* armv7a
* aarch64 (Common for Coral EdgeTPU devices.)

Optionally, the binary can be deployed to vanilla Coral device with the
following command:

```
scp bin/ondevice_demo_aarch64 mendel@192.168.100.2:~/ondevice_demo_aarch64
```

## Visualization

Running `./ondevice_demo_*` will create .txt files with the classes, score, and
bounding boxes. To visualize the output, you may run the following command
while still in the docker shell:

```
bazel run //tools:visualizer -- --image_path=`pwd`/output/00001.bmp \
  --result_path=`pwd`/output/00001.bmp.txt \
  --output_path=`pwd`/00001_visualized.bmp
```

Note that bazel does not execute from the working directory, so the paths are
prefixed with \`pwd\`. This is not necessary if an absolute path is given.
