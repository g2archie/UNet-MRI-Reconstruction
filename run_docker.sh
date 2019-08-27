#!/bin/bash 

MODEL_NAME='UNet2D2D'
SOURCE='/home/wentao/tensorflow_serving_model/models/'$MODEL_NAME
TARGET='/models/'$MODEL_NAME

docker run --runtime=nvidia -p 8501:8501 \
--mount type=bind,source=$SOURCE,target=$TARGET -e\
MODEL_NAME=$MODEL_NAME -t tensorflow/serving:latest-gpu &