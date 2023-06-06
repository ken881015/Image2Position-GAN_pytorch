#!/bin/bash +x

source ./env_setup.sh

# remove docker container
if [[ "$(sudo docker ps -a -q  --filter ancestor=$container_name)" != "" ]]; then
    sudo docker rm $(sudo docker ps -a -q  --filter ancestor=$container_name) -f 
fi

# remove docker image
if [[ "$(sudo docker images -q $tag)" != "" ]]; then
    sudo docker rmi $tag
fi

