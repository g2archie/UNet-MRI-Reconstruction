from os.path import basename
from os.path import join

import smtplib
import yaml
import h5py
import numpy
import socket
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication


def write_to_h5(file_name, result_dict):
    f = h5py.File(file_name, 'w')

    for k, v in result_dict.items():
        f[k] = v

    f.close()


def get_hostname():
    return socket.gethostname()


def extract_images(filename, image_name):
    data = h5py.File(filename, 'r').get(image_name)
    data = numpy.array(data)
    print("Shape of data is {}".format(data.shape))

    return data


def load_training_settings(file='./config/training_config.yaml'):
    with open(file, 'r') as f:
        return yaml.load(f, Loader=yaml.Loader)


def load_email_server_settings():
    with open('./config/email_config.yaml', 'r') as f:
        return yaml.load(f, Loader=yaml.Loader)


def send_email(subject, message, files=None):
    config = load_email_server_settings()

    email_server = smtplib.SMTP(config['smtp_server'], config['smtp_server_port'])
    email_server.starttls()

    try:
        email_server.login(config['sender_email_address'], config['sender_email_password'])
    except Exception as e:
        print("Cannot log in: {}".format(str(e)))
        return

    msg = MIMEMultipart()  # create a message

    msg['From'] = config['sender_email_address']
    msg['To'] = config['receiver_email_address']
    msg['Subject'] = subject

    if files is not None:
        for file in files:
            try:
                with open(file, 'r') as f:
                    part = MIMEApplication(f.read(), Name=basename(file))
                part['Content-Disposition'] = 'attachment; filename="{}"'.format(basename(file))
                msg.attach(part)
            except FileNotFoundError as e:
                print(e)
                return

    msg.attach(MIMEText(message, 'plain'))

    try:
        email_server.send_message(msg)
    except Exception as e:
        print("Cannot send email, error : {}".format(str(e)))
        return

    email_server.quit()


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size, shuffle=True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(x_set))

    def __len__(self):
        return np.math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indexes)

