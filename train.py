import os
import pickle
from datetime import datetime

from models.UNet3D import UNet3D
from models.UNet2D1D import UNet2D1D
from models.UNet2D2D import UNet2D2D
from models.UNet3D_old import UNet3D_old

from utils import send_email
from utils import write_to_h5
from utils import get_hostname
from utils import extract_images
from utils import load_training_settings

from loss.custom_loss import *
from metrics.custom_metrics import *

host_name = get_hostname()
NETWORK_TYPES = {'UNet3D': UNet3D, 'UNet2D1D': UNet2D1D, 'UNet3D_old': UNet3D_old, 'UNet2D2D': UNet2D2D}
OPTIMIZER_TYPES = {'Adam': tf.keras.optimizers.Adam, 'RMSprop': tf.keras.optimizers.RMSprop}
LOSS_TYPES = {'ssim_loss': ssim_loss, 'psnr_loss': psnr_loss}
METRICS_TYPES = {'ssim': ssim, 'psnr': psnr}
SEND_SUMMARY = True

tasks = load_training_settings()
no_of_tasks = len(tasks)

if SEND_SUMMARY:
    send_email('The task on {} has started, total tasks: {}.'.format(host_name, no_of_tasks),
               'Details of the training are in the attachment ', files=['./config/training_config.yaml'])

for index, task in enumerate(tasks):

    try:
        # Allow growth for GPU Memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # tf.keras.backend.set_floatx('float32')
        tf.keras.backend.set_session(sess)

        # Define Folders for the output
        task_name = task['task_name']
        task_type = task['task_type']
        OUTPUT_DIR = os.path.join(os.path.abspath(task['output_data_dir']), task_name,
                                  datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        tensorboard_logdir = os.path.join(OUTPUT_DIR, 'logs/')
        model_weights_dir = os.path.join(OUTPUT_DIR, 'model_weights/')
        tf_serving_model_dir = os.path.join(OUTPUT_DIR, 'tf_serving/')
        training_history_dict = os.path.join(OUTPUT_DIR, 'training_history')
        prediction_dir = os.path.join(OUTPUT_DIR, 'predictions/')

        checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoints/cp-{epoch:04d}-ssim-{val_ssim:.4f}.ckpt")
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Load neural network settings, set the batch size
        network_settings = task['network_settings']
        batch_size = network_settings['batch_size']

        # Load corresponding data depending on the type of the task
        if task_type in ['train', 'train_and_predict']:
            x_train = extract_images(task['input_data_path']['x_train'], 'imagesRecon')
            y_train = extract_images(task['input_data_path']['y_train'], 'imagesTrue')
            x_validation = extract_images(task['input_data_path']['x_val'], 'imagesRecon')
            y_validation = extract_images(task['input_data_path']['y_val'], 'imagesTrue')
            # input_data_shape = x_train.shape

        # Create callback list. Checkpoint, tensorbaord and earlystopping callback are added by default.
        callback_list = []
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_ssim', verbose=1,
                                                         save_weights_only=True)
        callback_list.append(cp_callback)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logdir, histogram_freq=2,
                                                              write_graph=True, write_grads=True, write_images=True,
                                                              batch_size=batch_size)
        callback_list.append(tensorboard_callback)
        if network_settings['early_stopping']['use']:
            earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_ssim',
                                                                      patience=network_settings['early_stopping'][
                                                                          'patience'],
                                                                      min_delta=network_settings['early_stopping'][
                                                                          'min_delta'],
                                                                      mode='max'
                                                                      )
            callback_list.append(earlystopping_callback)

        # Create the optimizer and the model from the configuration file.
        optimizer = OPTIMIZER_TYPES[network_settings['optimizer']['type']](
            network_settings['optimizer']['learning_rate'])

        model = NETWORK_TYPES[network_settings['network_type']](network_settings['regularization']['type'],
                                                                network_settings['regularization']['parameters'])
        # Set the loss and the metrics from the configuration file.
        loss_name = network_settings['loss']
        if loss_name in LOSS_TYPES:
            loss = LOSS_TYPES[loss_name]
        else:
            loss = loss_name

        metrics = []
        for metric_name in network_settings['metrics']:
            if metric_name in METRICS_TYPES:
                metrics.append(METRICS_TYPES[metric_name])
            else:
                metrics.append(metric_name)

        # Compile and build the model.

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

        start_notification = 'The {} task  of {} has been set up and ready for training on {} task {} of {}.'.format(
            task_type,
            task_name,
            host_name,
            index + 1,
            no_of_tasks)
        print(start_notification)

        if task['email_notification']:
            send_email(start_notification,
                       'Please see the details of settings in the previous email')

        # Fit the model and save the model weights, the pb file for tensorflow serving and the training history dict
        # into a pickle file.

        if task_type in ['train', 'train_and_predict']:
            history = model.fit(x_train, y_train, batch_size=batch_size,
                                epochs=network_settings['epochs'],
                                validation_data=((x_validation, y_validation)),
                                callbacks=callback_list, verbose=2)

            model.save_weights(model_weights_dir, save_format='tf')
            tf.keras.experimental.export_saved_model(model, tf_serving_model_dir, serving_only=True)

            with open(training_history_dict, 'wb') as file_pi:
                pickle.dump(history.history, file_pi)

        # Load the network weight
        if task_type == 'predict':
            saved_network_weights_path = task['input_data_path']['saved_network_weights']
            model.load_weights(saved_network_weights_path)

        # Predict the x_test and save the result into a h5 file for later visualization
        if task_type in ['predict', 'train_and_predict']:
            x_test = extract_images(task['input_data_path']['x_test'], 'imagesRecon')
            y_test = extract_images(task['input_data_path']['y_test'], 'imagesTrue')

            result = model.predict(x_test, batch_size=batch_size, verbose=0)
            result_dict = {
                'input': x_test,
                'result': result,
                'truth': y_test
            }
            os.makedirs(prediction_dir)
            write_to_h5(prediction_dir + 'result.h5', result_dict)
        end_notification = 'The {} task of {} has ended on {}, task {} of {}.'.format(task_type, task_name, host_name,
                                                                                      index + 1,
                                                                                      no_of_tasks)
        print(end_notification)
        if task['email_notification']:
            send_email(end_notification,
                       'Please see the details of settings in the previous email')

    except Exception as e:
        error_notification = "When completing the task {} of {}, error: {}, task stopped. ".format(index + 1,
                                                                                                   no_of_tasks,
                                                                                                   str(e))
        print(error_notification)
        send_email('The task on {} has stopped due to error, {} of {}'.format(host_name, index + 1, no_of_tasks),
                   error_notification)
    finally:
        tf.keras.backend.clear_session()

if SEND_SUMMARY:
    send_email('The task on {} has ended, total trainings: {}.'.format(host_name, no_of_tasks),
               'Please login to see the output', files=['./nohup.log'])
