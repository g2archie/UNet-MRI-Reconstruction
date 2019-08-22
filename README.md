# UNet MRI Reconstruction
## Introduction
Classic Cardiovascular Magnetic Resonance takes a long time to obtain images over multiple hear beats. Real-time CMR is faster, but the data acquired is often of low spatial and temporal resolution. In this project, three CNNs are used to produce MRI image reconstruction, namely, UNet3D, UNet2D1D, and UNet2D2D.  

The configuration file training_config.yaml allows users to configure the training tasks and the hyperparameters in the network.  Tensorboard logs and Tensorflow serving models are produced in the output folder by default. 

![The input image](https://raw.githubusercontent.com/g2archie/UNet-MRI-Reconstruction/master/sample_images/UNet2D2D_no_regularization/input.jpg)

![The output image](https://raw.githubusercontent.com/g2archie/UNet-MRI-Reconstruction/master/sample_images/UNet2D2D_no_regularization/result.jpg)

![The truth image](https://raw.githubusercontent.com/g2archie/UNet-MRI-Reconstruction/master/sample_images/UNet2D2D_no_regularization/truth.jpg)
## Installation

### Train your own models
Install [CUDA Toolkit 10.0]([https://developer.nvidia.com/cuda-10.0-download-archive?](https://developer.nvidia.com/cuda-10.0-download-archive?)) as this is require for Tensorflow-gpu==1.14.0

Create an Anaconda environment, then in the activated environment, run
```
pip install -r requirements.txt
```
This should install all the dependencies required for training the model. 

### Docker installation
If you want to use docker to serve your model or use my pre-trained model. Please install [Docker]([https://docs.docker.com/install/](https://docs.docker.com/install/))  and [Nvidia-Docker]([https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)) on your machine.

### Use Docker images

I will provide a docker TensorFlow serving images which contains my best pre-trained model.

## Training Configuration 

A typical training task will look like below.
```yaml
- task_name: 'UNet2D2D_MSE_loss_no_regularization'
  task_type: 'train_and_predict'
  email_notification: true
  input_data_path:
    x_train: './data/input_train.h5'
    y_train: './data/truth_train.h5'
    x_test: './data/input_test.h5'
    y_test: './data/truth_test.h5'
    x_val: './data/input_validation.h5'
    y_val: './data/truth_validation.h5'
    saved_network_weights:
  network_settings:
    network_type: 'UNet2D2D'
    batch_size: 1
    epochs: 20
    loss: 'MSE'
    metrics:
              - 'ssim'
    optimizer:
      type: 'Adam'
      learning_rate: 1.0e-04
    regularization:
      type: 'l2'
      parameters:
                 - 1.0e-05
    early_stopping:
      use: true
      patience: 2
      min_delta: 1.0e-02
  output_data_dir: './keras_training'
```

* task_name is used in notifications and constructing the output folder.
* task_type has three available options: 'train', which trains and store the model but not predicting the x_test. 'train_and_predict', which also produces a result.h5 file after prediction. 'predict', which only loads the model's weights and perform prediction.
* email_notification If this is set to True, you will receive email notifications at the beginning of the job and *each task*, when any exception triggered, the end of *each task* and the job.
* input_data_path: These are for the input data paths.
* network_type:  It specifies the model to use in the task, available options are UNet3D, UNet2D1D, UNet2D2D and UNet3D_old.
* batch_size:  Network hyperparameters.  
* epochs:  Network hyperparameters.  
* loss:  It can be set to any losses by Keras in a string or my custom losses, 'psnr_loss' and 'ssim_loss'.
* metrics:  It expects a list of strings, again, any metrics by Keras can be used, and my custom metrics, 'ssim' and 'psnr' are also avaiable.
* optimizer:  type:  By now only two optimizers are implemented, 'Adam' and 'RMSprop'
* optimizer:  learning_rate:  Optimizer paramenter.
* regularization:  type:  Now, there are five options. 'l2', 'l1', 'l1_l2', 'batch_norm', and 'instance_norm', 'dropout' and 'dropblock'.
* regularization:  parameters: It expects a list of values, for example for 'dropblock',  the parameters are keep_prob and block_size, so [0.8, 3] should be set in YAML format.  **Note** that these layers are added after all convolutional layers by default.
* early_stopping: use: Set to true if you want to use the early stopping callback.
* early_stopping: patience:    early_stopping parameter
* early_stopping: min_delta:  early_stopping parameter
* output_data_dir:  The directory where all the output data is stored.

It supports running the tasks consecutively so that you can add more than one tasks in the configuration file.

## Email Configuration
If you want to enable the email notification, please modify the file
```
 ./config/sample_email_config.yaml 
```
then rename it to 
```
email_config.yaml
```
## Run the training
You can either activate your virtual environment and run 
```
conda activate python3.7
python train.py
``` 
or modify the bash script train.sh to activate your virtual environment then run the training so that you can type 
```
./train.sh
```
You can also specify the GPUs you want to use in the bash script file. I have not added tf.distribute.MirroredStrategy due to I has only got one GPU now.

My machine has 16 GB RAM and RTX 2060 8GB graphics card and most of the traininig can be run under such setup.

If you want to drop the SSH connection after you set up the job, run

```
./nohup.sh
``` 
## Output files
The output path looks like this by default.
```
./keras_training/UNet2D2D_SSIM_loss_no_regularization/20190818-075223/
```
It is just output_data_dir/task_name/time/.

If you run a 'train_and_predict' task by default, you will get 
* tf_serving folder which contains the model that can be used by Tensorflow serving.
* predictions folder which includes a single h5 file named result.h5 which contains the prediction result.
* model_weights folder which contains the weights of the final trained model.
* logs folder which contains logs created by Tensorboard callback, which can be viewed by Tensorboard.
* checkpoints folder, which contains the weights of the model after each epoch.
* training_history pickle file, which contains the loss of the training in a python dictionary.

## Visulization of the result

### Tensorboard

Run the following command in 
```
tensorboard --_logdir_=/path/to/the/log/dir
```

### Jupyter notebook

I will provide a notebook which allows to visualize the result and calculates stats like MSE, SSIM and PSNR.