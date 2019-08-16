import tensorflow as tf


def ssim_loss(y_true, y_pred):

    ssim = tf.image.ssim(y_true, y_pred, max_val=1)

    return tf.math.subtract(tf.constant(1.0), ssim)


def psnr_loss(y_true, y_pred):

    psnr = tf.image.psnr(y_true, y_pred, max_val=1)

    return -psnr
