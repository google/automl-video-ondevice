# Prerequisites

nvidia docker is pre-installed through JetPack after 4.4. 

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

# Running a TFTRT Model using DeepStream

Make sure you are in the top-most `automl-video-ondevice` directory.

Grant local user access to X Screen (for displaying outputs from DeepStream):

`$ xhost +`

Retrieve and start the docker container:

`docker run -it --rm --net=host --runtime nvidia  -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-5.0 \
-v $(pwd):/automl-video-ondevice -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/deepstream-l4t:5.0-20.07-samples`

We also prepared the `run_ds_docker.sh` script that you can run instead.The script will grant an access to X screen and start the docker container. 

For non-Jetson devices, such as a Desktop replace nvcr.io/.../...-base with
`nvcr.io/nvidia/deepstream:5.0-20.07-samples` instead.

Note: If you don't want to use Docker, you may install DeepStream directly.
[Follow the guide provided by NVIDIA to do so.](https://docs.nvidia.com/metropolis/deepstream/5.0/dev-guide/index.html)

### a) Output to an X-window:

`$ cd /automl-video-ondevice`

`$ deepstream-app -c deepstream/automl_ds_x11_pipeline.txt`

###  b) Output to a video file.

`$ cd /automl-video-ondevice`

`$ deepstream-app -c deepstream/automl_ds_file_pipeline.txt`

You can edit automl_ds_file_pipeline.txt to specify the output destination.


# Running a TRT Model using DeepStream

TRT Model files are placed in `/data` folder :
`traffic_model_fp32_nano.plan` (generated on Jetson Nano, 7.1.3 TRT version) 
`traffic_model_fp16_xavier_nx.plan`(generated on Jetson Xavier NX , 7.1.3 TRT version) 

Depending on the platform you use to run the model copy the specific model to `deepstream/vot/1/`

`config.pbtxt.trt` in `deepstream/vot` directory  is the file that will be used for deepstrem-app to run with an engine file. The last line in this file refers to the name of the engine file you will be using . In our example we want DeepStream to run TRT model using Jetson Xavier NX. In this case, config.pbtxt.trt should call the right model.  

`default_model_filename: "traffic_model_fp16_xavier_nx.plan"`

To enable DeepStream to use this config file instead of the default one, copy the config.pbtxt.trt to config.pbtxt
`$ cp config.pbtxt.trt config.pbtxt`

Again, you can run the DeepStream in two ways using the following commands:

### a) Output to an X-window:
`$ cd /automl-video-ondevice`

`$ deepstream-app -c deepstream/automl_ds_trt_x11_pipeline.txt`

###  b) Output to a video file.

`$ cd /automl-video-ondevice`

`$ deepstream-app -c deepstream/automl_ds_trt_file_pipeline.txt`


