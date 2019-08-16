#!/bin/bash 

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=1

ANACONDA_BIN_PATH=/home/wentao/anaconda3/bin

echo "> Loading Anaconda 'python3.7' environment..."
source $ANACONDA_BIN_PATH/activate python3.7
python --version

echo "> Environment loaded."
echo "> Runnning training script..."
python train.py
echo "> Training script finished."
echo "> Deactivating Anaconda environment..."
conda deactivate
echo "> Anaconda environment deactivated"
