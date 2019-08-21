# AutoML Video On-Device Examples

The example code that shows how to load the Google Cloud AutoML Video Object
Tracking On-Device models and conduct inference on a sequence of images from a
video clip.

The targeted devices are CPU and Edge TPU.

## Maintainers
* Yongzhe Wang (yongzhe@google.com)
* Henry Quoc Tran (henryquoctran@google.com)

# Building
Launch a docker shell.
```
make -f ondevice-examples.makefile  docker-example-shell
```

Enter working directory.
```
cd /edgetpu-ml-cpp
```

Build the binaries. For development, stay in the docker shell and re-run the
following command to create a new build.
```
make -f ondevice-examples.makefile ondevice-examples
```

The resulting binaries will be copied to `cpp_example_out/ondevice_demo_*`.

The ondevice-examples command will build binaries for amd64 (desktop), arm64,
and arm32.

For faster development, use a platform specific makefile rule:
```
make -f ondevice-examples.makefile ondevice-examples-arm64
```

# Visualization

Running `./ondevice_demo` will create .txt files with the classes, score, and bounding boxes. To visualize the output, you may run the following command while still in the docker shell:

```
bazel run //tools:visualizer -- --image_path=`pwd`/output/00001.bmp --result_path=`pwd`/output/00001.bmp.txt --output_path=`pwd`/00001_visualized.bmp
```

Note that bazel does not execute from the working directory, so the paths are prefixed with \`pwd\`. This is not necessary if an absolute path is given.
