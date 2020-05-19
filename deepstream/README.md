# Prerequisites

`sudo apt-get install -y gcc build-essential docker.io`

Install nvidia-docker: https://github.com/NVIDIA/nvidia-docker

Clone this repo: `git clone https://github.com/google/automl-video-ondevice`

# Preparing your AutoML Video models for DeepStream

Note: If you want to just try running DeepStream using the sample models, you
may skip to the next section.

Note: Python3.7 is required to run these steps.

Generate a TRT optimized version of the model downloaded from AutoML Video. It
is important to run this command on the hardware that you want the optimizations
for:

`$ python3 tools/trt_compiler.py --model=your_model.pb`

This tool should output the generated model to `{model_name}_trt.pb`

Replace `deepstream/vot/1/frozen_inference_graph_trt.pb` with the TRT-optimized
model:

`$ mv your_model_trt.pb deepstream/vot/1/frozen_inference_graph_trt.pb`

Next, convert the AutoML generated label map into the DeepStream format:

`$ python3 tools/ds_label_map_converter.py --label_map=your_label_map.pbtxt
--output=deepstream/vot/label.txt`

Finally, open `deepstream/vot/automl_ds_detect.txt` and change
`num_detected_classes: 4` to the number of classes in your label map.

You can figure out the number of classes in the label map by running:

```expr `wc -l < deepstream/vot/label.txt` - 1```

# Running a model using DeepStream

Make sure you are in the top-most `automl-video-ondevice` directory.

Grant local user access to X Screen (for displaying outputs from DeepStream):

`$ xhost +`

Retrieve and start the docker container:

`$ docker run --gpus all -it --rm -v $(pwd):/automl-video-ondevice -v
/tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY
nvcr.io/nvidia/deepstream-l4t:5.0-dp-20.04-base`

For non-Jetson devices, such as a Desktop replace nvcr.io/.../...-base with
`nvcr.io/nvidia/deepstream:5.0-dp-20.04-triton` instead.

Note: If you don't want to use Docker, you may install DeepStream directly.
[Follow the guide provided by NVIDIA to do so.](https://docs.nvidia.com/metropolis/deepstream/5.0/dev-guide/index.html)

## a) Output to an X-window:

`$ cd /automl-video-ondevice`

`$ deepstream-app -c deepstream/automl_ds_x11_pipeline.txt`

##  b) Output to a video file.

`$ cd /automl-video-ondevice`

`$ deepstream-app -c deepstream/automl_ds_file_pipeline.txt`

You can edit automl_ds_file_pipeline.txt to specify the output destination.
