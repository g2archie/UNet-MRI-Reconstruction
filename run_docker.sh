#!/bin/bash 


SOURCE='/home/wentao/tensorflow_serving_model/models/UNet2D2D'
TARGET='/models/UNet2D2D'
MODEL_NAME='UNet2D2D'


docker run --runtime=nvidia -p 8501:8501 --mount type=bind,source=$SOURCE,target=$TARGET -e MODEL_NAME=$MODEL_NAME -t tensorflow/serving:latest-gpu &



