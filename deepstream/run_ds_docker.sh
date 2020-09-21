xhost +
sudo docker run -it --rm --net=host --runtime nvidia  -e DISPLAY=$DISPLAY -w /opt/nvidia/deepstream/deepstream-5.0 \
-v $(pwd):/automl-video-ondevice -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/deepstream-l4t:5.0-20.07-samples
