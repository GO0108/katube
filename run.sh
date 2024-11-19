#!/bin/bash

echo -e "\nAbrindo container."
sudo docker run \
        -it \
        --rm \
        --ipc=host \
        --volume "/home/aluno_marceloferreira":"/app" \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --memory-swap -1 \
        --name 'ermis-dataset' \
        --gpus all \
        katube-gamma

