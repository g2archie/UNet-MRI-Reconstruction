import tensorflow as tf

from layers.instancenormalization import InstanceNormalization
from layers.dropblock import DropBlock3D


class UNet3D(tf.keras.Model):

    def __init__(self, regularization=None, regularization_parameters=None):
        super().__init__()
        self.depth = 5
        self.regularization = regularization
        self.regularization_parameters = regularization_parameters
        self.kernel_regularizer = None
        if regularization is not None:
            if regularization == 'l2':
                self.kernel_regularizer = tf.keras.regularizers.l2(*regularization_parameters)
            elif regularization == 'l1':
                self.kernel_regularizer = tf.keras.regularizers.l1(*regularization_parameters)
            elif regularization == 'l1_l2':
                self.kernel_regularizer = tf.keras.regularizers.l1_l2(*regularization_parameters)

        self.conv3d_1_1 = tf.keras.layers.Conv3D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3d_1_1',
                                                 kernel_regularizer=self.kernel_regularizer)

    def _add_regularization_layer(self, input_layer, name_suffix):

        if self.regularization == 'batch_norm':
            layer_name = "Batch_Norm_" + name_suffix
            if hasattr(self, layer_name):
                batch_norm_layer = getattr(self, layer_name)
            else:
                batch_norm_layer = tf.keras.layers.BatchNormalization(-1, name=layer_name)
                setattr(self, layer_name, batch_norm_layer)
            return batch_norm_layer(input_layer)

        elif self.regularization == 'instance_norm':
            layer_name = "Instance_Norm_" + name_suffix
            if hasattr(self, layer_name):
                instance_norm_layer = getattr(self, layer_name)
            else:
                instance_norm_layer = InstanceNormalization(-1, name=layer_name)
                setattr(self, layer_name, instance_norm_layer)
            return instance_norm_layer(input_layer)

        elif self.regularization == 'dropout':
            layer_name = "Dropout_" + name_suffix
            if hasattr(self, layer_name):
                dropout_layer = getattr(self, layer_name)
            else:
                dropout_layer = tf.keras.layers.Dropout(*self.regularization_parameters, name=layer_name)
                setattr(self, layer_name, dropout_layer)
            return dropout_layer(input_layer)
        elif self.regularization == 'dropblock':
            layer_name = "DropBlock_" + name_suffix
            if hasattr(self, layer_name):
                dropblock_layer = getattr(self, layer_name)
            else:
                dropblock_layer = DropBlock3D(*self.regularization_parameters, name=layer_name)
                setattr(self, layer_name, dropblock_layer)
            return dropblock_layer(input_layer)

        return input_layer

    def _get_convolution_block(self, input_layer, filters, kernel_size=3, strides=1, padding='same',
                               name_prefix='l_', activation=tf.keras.activations.relu):

        conv3d_layer_name_1 = name_prefix + "Conv3D_{}_1".format(filters)

        if hasattr(self, conv3d_layer_name_1):
            conv3d_1 = getattr(self, conv3d_layer_name_1)
        else:
            conv3d_1 = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                              name=conv3d_layer_name_1, activation=activation,
                                              kernel_regularizer=self.kernel_regularizer, data_format='channels_last')

            setattr(self, conv3d_layer_name_1, conv3d_1)

        conv3d_1 = conv3d_1(input_layer)
        conv3d_1 = self._add_regularization_layer(conv3d_1, name_suffix=conv3d_layer_name_1)

        conv3d_layer_name_2 = name_prefix + "Conv3D_{}_2".format(filters)

        if hasattr(self, conv3d_layer_name_2):
            conv3d_2 = getattr(self, conv3d_layer_name_2)
        else:
            conv3d_2 = tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                              name=conv3d_layer_name_2, activation=activation,
                                              kernel_regularizer=self.kernel_regularizer, data_format='channels_last')

            setattr(self, conv3d_layer_name_2, conv3d_2)

        conv3d_2 = conv3d_2(conv3d_1)
        conv3d_2 = self._add_regularization_layer(conv3d_2, name_suffix=conv3d_layer_name_2)

        return conv3d_2

    def _get_convolution_transpose_layer(self, input_layer, filters, kernel_size=3, strides=(2, 2, 1), padding='same',
                                         name_prefix='r_', activation=tf.keras.activations.relu):

        conv3d_transpose_layer_name = name_prefix + "UpConv3D_{}".format(filters)
        if hasattr(self, conv3d_transpose_layer_name):
            conv3d_transpose = getattr(self, conv3d_transpose_layer_name)
        else:
            conv3d_transpose = tf.keras.layers.Convolution3DTranspose(filters=filters, kernel_size=kernel_size,
                                                                      strides=strides, padding=padding,
                                                                      activation=activation,
                                                                      name=conv3d_transpose_layer_name,
                                                                      kernel_regularizer=self.kernel_regularizer)

            setattr(self, conv3d_transpose_layer_name, conv3d_transpose)

        conv3d_transpose = conv3d_transpose(input_layer)
        conv3d_transpose = self._add_regularization_layer(conv3d_transpose, name_suffix=conv3d_transpose_layer_name)
        return conv3d_transpose

    def _get_max_pool_3d_layer(self, filters, pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same',
                               name_prefix='l_'):

        maxpool_3d_layer_name = name_prefix + "MaxPool3D_{}".format(filters)

        if hasattr(self, maxpool_3d_layer_name):
            maxpool_3d = getattr(self, maxpool_3d_layer_name)
        else:
            maxpool_3d = tf.keras.layers.MaxPooling3D(pool_size=pool_size, strides=strides, padding=padding,
                                                      name=maxpool_3d_layer_name)
            setattr(self, maxpool_3d_layer_name, maxpool_3d)

        return maxpool_3d

    def call(self, inputs):

        inputs = tf.keras.backend.expand_dims(inputs, axis=-1)
        current_layer = inputs
        level = []

        for depth in range(self.depth):
            filters = 32 * 2 ** depth
            first_layer = self._get_convolution_block(input_layer=current_layer, filters=filters)
            if depth < self.depth - 1:
                current_layer = self._get_max_pool_3d_layer(filters=filters)(first_layer)
                level.append([first_layer, current_layer])
            else:
                current_layer = first_layer
                level.append([first_layer])

        for depth in range(self.depth - 2, -1, -1):
            filters = 32 * 2 ** depth
            up_convolution_layer = self._get_convolution_transpose_layer(input_layer=current_layer, filters=filters,
                                                                         kernel_size=2)
            concat_layer = tf.keras.layers.concatenate([level[depth][0], up_convolution_layer], axis=-1)
            current_layer = self._get_convolution_block(input_layer=concat_layer, filters=filters, name_prefix='r_')

        conv3d_1_1 = self.conv3d_1_1(current_layer)
        conv3d_1_1 = self._add_regularization_layer(conv3d_1_1, name_suffix='conv3d_1_1')

        sum = tf.keras.layers.add([conv3d_1_1, inputs])
        update = tf.keras.activations.relu(sum)
        output = tf.keras.backend.squeeze(update, -1)
        return output
