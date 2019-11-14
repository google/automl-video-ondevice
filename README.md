# AutoML Video On-Device Examples

The example code that shows how to load the Google Cloud AutoML Video Object
Tracking On-Device models and conduct inference on a sequence of images from a
video clip.

The targeted devices are CPU and Edge TPU.

---

## Dependencies

Docker (>= 18.09.3)

Docker is used to simplify the development process and avoid any
environment and setup issues.

## Getting the Code

```
git clone https://github.com/google/automl-video-ondevice
git submodule update --init --recursive
```

## Building Examples

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

## Running

### For Linux Desktop


```
./bin/k8/ondevice_demo --alsologtostderr \
  --model_file_path=./data/traffic_model.tflite \
  --label_map_file_path=./data/traffic_label_map.pbtxt \
  --images_file_path=./data/traffic_frames
```

### For Coral Dev Board

The binary and test data must be deployed to the device:

```
scp bin/aarch64/ondevice_demo mendel@192.168.100.2:~/ondevice_demo_aarch64
scp -r data mendel@192.168.100.2:~/data
```

Then SSH into the device:

```
ssh mendel@192.168.100.2
```

Finally, running inference:

```
cd ~
./ondevice_demo_aarch64 --alsologtostderr \
  --model_file_path=./data/traffic_model_edgetpu.tflite \
  --label_map_file_path=./data/traffic_label_map.pbtxt \
  --images_file_path=./data/traffic_frames
```

The output can be pulled out with the following command:

```
scp -r mendel@192.168.100.2:~/data /tmp/coral-output
```

## Visualization

Running `./ondevice_demo_*` will create .txt files with the classes, score, and
bounding boxes. To visualize the output, you may run the following command
while still in the docker shell (or on the host if you have bazel installed):

```
bazel run //tools:visualizer -- \
  --image_path="`pwd`/data/traffic_frames/*.bmp" \
  --output_path=/tmp/output
```

Note that bazel does not execute from the working directory, so the paths are
prefixed with \`pwd\`. This is not necessary if an absolute path is given.
