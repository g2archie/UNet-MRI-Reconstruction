#!/bin/bash 
#
# Following script will:
# * Load  Anaconda 'python3.7' environmnet
# * Call unet2D1D_main.py with proper arguments in order to train the network.
# * Deactivate Anaconda 'python3.7' environment
# * Clear __pycache__
#
# NOTE: the script must be called using 'source' in order to load it to your curent shell.
# NOTE:  arguments for unet2D1D_main.py are quite redundant and are about
#       to be changed as soon as possible.
# NOTE: it is not safe to move the process to the background with suspending it and using
#       its job number. 

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

ANACONDA_BIN_PATH=/home/wentao/anaconda3/bin
MODE="training"
#TRAIN_DATA_PATH_RECON="/home/wentao/bartek/MRdata/trainData_half.h5"
#TRAIN_DATA_PATH_RECON="/home/wentao/bartek/MRdata/testData_tga_rot_fullSet.mat"
TRAIN_DATA_PATH_RECON="/home/wentao/bartek/MRdata/trainData_tga_rot.mat"

#TRAIN_DATA_PATH_TRUE="/home/wentao/bartek/MRdata/trainData_truth_half.h5"
#TRAIN_DATA_PATH_TRUE="/home/wentao/bartek/MRdata/testData_truth.mat"
TRAIN_DATA_PATH_TRUE="/home/wentao/bartek/MRdata/trainData_truth.mat"
#OUTPUT_DATA_PATH="network_128_by_128/full_dataset_unet3D_2.cpkt"
#SAVED_NETWORK_PATH="network_128_by_128/full_dataset_unet3D_2.cpkt"
SAVED_NETWORK_PATH="network_128_by_128/full_train_ssim_loss_with_l2_regularization_2.cpkt"
OUTPUT_DATA_PATH="network_128_by_128/full_train_ssim_loss_with_l2_regularization_2.cpkt"


echo "> Loading Anaconda 'python3.7' environment..."
source $ANACONDA_BIN_PATH/activate python3.7
python --version
echo "> Environment loaded."
echo "> Runnning training script..."
echo "python3 unet3D_main.py "$MODE" "$TRAIN_DATA_PATH_RECON" "$TRAIN_DATA_PATH_TRUE" "$OUTPUT_DATA_PATH" "$SAVED_NETWORK_PATH""
python3 -u unet3D_main.py "$MODE"  "$TRAIN_DATA_PATH_RECON" "$TRAIN_DATA_PATH_TRUE" "$OUTPUT_DATA_PATH" "$SAVED_NETWORK_PATH"
echo "> Training script finished."
echo "> Deactivating Anaconda environment..."
conda deactivate
echo "> Anaconda environment deactivated"
echo "> Clearing pycache..."
rm -rf __pycache__
echo "> Python's cache cleared."
