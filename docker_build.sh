#!/bin/bash

source env_setup.sh

# # build docker image
if [[ "$(sudo docker images -q $tag)" == "" ]]; then
    sudo docker build --build-arg UID=1000 --build-arg GID=1000 --build-arg NAME=$GIT_NAME  -t $tag -f ./Docker/Dockerfile ./Docker
fi

bash ./docker_run.sh

