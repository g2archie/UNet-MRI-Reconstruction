import tensorflow as tf


def ssim(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1)


def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1)
