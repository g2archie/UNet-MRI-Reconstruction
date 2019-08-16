from os.path import basename
import smtplib
import yaml
import h5py
import numpy
import socket
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
    except:
        print("Cannot log in")
        return

    msg = MIMEMultipart()       # create a message

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
    except:
        print("Cannot send email")
        return

    email_server.quit()


