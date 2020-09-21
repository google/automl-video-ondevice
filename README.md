# AutoML Video Edge Library

AutoML Video Edge Library is an open source engine used for inferencing models
trained using AutoML Video. It supports running Tensorflow, TF-TRT, TFLite, and
EdgeTPU-optimized TFLite models.

I'm Developing For:

* [Linux Desktop](#for-linux-desktop)
* [Coral Device](#for-coral-device)
* [NVIDIA Jetson](#for-nvidia-jetson)

# For Linux Desktop
-------------------

If you are looking to do inferencing with no additional hardware, using only CPU
then you may use the vanilla Tensorflow (.pb) and TFLite (.tflite) models.

## Prerequisites

```
sudo apt-get update
sudo apt-get install python3.7
sudo apt-get install python3-pip
pip3 install opencv-contrib-python --user
pip3 install numpy
```

Note: opencv-contrib-python is only necessary for the examples, but can be
excluded if only the library is being used.

If you plan on running TFLite models on the desktop, install the TFLite
interpreter: https://www.tensorflow.org/lite/guide/python

If you plan on running Tensorflow models on desktop:  
`pip3 install tensorflow==1.14`

## Get the Code

`git clone https://github.com/google/automl-video-ondevice`

After that is done downloading, move into the directory.  
`cd automl-video-ondevice`

## Running an Example

For TFLite:  
`python3 examples/video_file_demo.py --model=data/traffic_model.tflite`

For Tensorflow:  
`python3 examples/video_file_demo.py --model=data/traffic_model.pb`

# For Coral Device
-------------------

## Prerequisites

Make sure you've setup your coral device:
https://coral.ai/docs/setup

Install the TFLite runtime on your device:
https://www.tensorflow.org/lite/guide/python

```
sudo apt-get update
sudo apt-get install git
sudo apt-get install python3-opencv
pip3 install numpy
```

## Get the Code

`git clone https://github.com/google/automl-video-ondevice`

After that is done downloading, move into the directory.  
`cd automl-video-ondevice`

## Running an Example

`python3 examples/video_file_demo.py --model=data/traffic_model_edgetpu.tflite`

# For NVIDIA Jetson
-------------------

## Prerequisites

```
sudo apt-get update
sudo apt-get install git
sudo apt-get install python3.7-pip
sudo apt-get install python3-opencv
sudo apt-get install protobuf-compiler libprotoc-dev
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
pip3 install cython numpy
pip3 install --no-cache-dir --ignore-installed --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 'tensorflow<2'

git clone https://github.com/NVIDIA/TensorRT.git 
cd oss/tools/onnx-graphsurgeon
make install

pip3 install --ignore-installed git+https://github.com/onnx/tensorflow-onnx
```

## Get the Code

`git clone https://github.com/google/automl-video-ondevice`

After that is done downloading, move into the directory.  
`cd automl-video-ondevice`

## Run a TensorFlow-TensorRT Example

`python3 examples/video_file_demo.py --model=data/traffic_model_trt.pb`

## Run a TensorRT Example
For TensorRT, the chip Engines converted with must be the same kind of chip to run inference workflow on. We hereby convert a TensorRT Engine on the Nano Platform. Later on, we hope to export the onnx model from Cloud console to hide the onnx converting steps from users.

First convert the model to ONNX format, use the model downloaded from GCP Video Intelligence console:
`python3 -m tf2onnx.convert --graphdef frozen_inference_graph.pb --inputs image_tensor:0 --outputs detection_boxes:0,detection_classes:0,detection_scores:0,num_detections:0 --output models/model_opset_10.onnx --fold_const --opset 10 --verbose`

then, 
`python3 examples/modify_onnx_model_opset10.py --onnx model_opset_10.onnx --modified model_opset_10_modified.onnx`

Then convert ONNX graph to a TensorRT Engine file, please note the TensorRT conversion is a memory intensive process so be mindful when running on less powerful device like Nano:
`/usr/src/tensorrt/bin/trtexec --onnx=model_opset_10_modified.onnx --fp16 --workspace=2048 --allowGPUFallback --saveEngine=vot_fp16.plan --verbose &> trtexec_build.log`

The log could be viewed by `tail -f trtexec_build.log`. The TensorRT Engine file `vot_fp16.plan` could be deployed to a Deep Stream application.


