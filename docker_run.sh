#!/bin/bash

source ./env_setup.sh

Image_id=$(sudo docker images -q $tag)

sudo docker run -it \
                --gpus "device=1" \
                --name  $container_name \
                --workdir $PWD \
                -v /srv:/srv \
                -v /tmp/.X11-unix:/tmp/.X11-unix \
                -e DISPLAY=$DISPLAY \
                --env QT_X11_NO_MITSHM=1 \
                -p 8086:22 \
                $Image_id

# -v $PWD:$PWD \
# port 22 for ssh, port 6006 for tensorboard

# -p 8083:6006 \