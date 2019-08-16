import tensorflow as tf

from layers.instancenormalization import InstanceNormalization
from layers.dropblock import DropBlock2D
from layers.dropblock import DropBlock3D


class UNet2D2D(tf.keras.Model):

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

    def _add_regularization_layer(self, input_layer, name_suffix, input_type='2d'):

        if self.regularization == 'batch_norm':
            layer_name = "Batch_Norm_" + name_suffix
            batch_norm_layer = tf.keras.layers.BatchNormalization(-1, name=layer_name)
            setattr(self, layer_name, batch_norm_layer)
            return batch_norm_layer(input_layer)

        elif self.regularization == 'instance_norm':
            layer_name = "Instance_Norm_" + name_suffix
            instance_norm_layer = InstanceNormalization(-1, name=layer_name)
            setattr(self, layer_name, instance_norm_layer)
            return instance_norm_layer(input_layer)

        elif self.regularization == 'dropout':
            layer_name = "Dropout_" + name_suffix
            dropout_layer = tf.keras.layers.Dropout(*self.regularization_parameters, name=layer_name)
            setattr(self, layer_name, dropout_layer)
            return dropout_layer(input_layer)

        elif self.regularization == 'dropblock':
            if input_type == '1d':
                return input_layer
            elif input_type == '2d':
                layer_name = "DropBlock_" + name_suffix
                dropblock_layer = DropBlock2D(*self.regularization_parameters, name=layer_name)
                setattr(self, layer_name, dropblock_layer)
                return dropblock_layer(input_layer)
            elif input_type == '3d':
                layer_name = "DropBlock_" + name_suffix
                dropblock_layer = DropBlock3D(*self.regularization_parameters, name=layer_name)
                setattr(self, layer_name, dropblock_layer)
                return dropblock_layer(input_layer)

        return input_layer

    def _get_convolution_block(self, input_layer, filters, kernel_size=3, strides=1, padding='same',
                               name_prefix='l_', activation=tf.keras.activations.relu):

        in_b, in_w, in_h, in_t, in_c = input_layer.get_shape().as_list()
        perm_input_tensor = tf.transpose(input_layer, perm=[0, 2, 1, 3, 4])
        reshaped_input_tensor = tf.reshape(perm_input_tensor, shape=(in_b * in_h,
                                                                     in_w, in_t,
                                                                     in_c))
        conv2d_layer_name_1 = name_prefix + "Conv2D_{}_1".format(filters)
        conv2d_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                          name=conv2d_layer_name_1, activation=activation,
                                          kernel_regularizer=self.kernel_regularizer, data_format='channels_last')
        setattr(self, conv2d_layer_name_1, conv2d_1)

        conv2d_1 = conv2d_1(reshaped_input_tensor)
        conv2d_1 = self._add_regularization_layer(conv2d_1, name_suffix=conv2d_layer_name_1)

        reshaped_output_tensor = tf.reshape(conv2d_1, shape=(in_w, in_t,
                                                             in_b, in_h, filters))

        perm_output_tensor = tf.transpose(reshaped_output_tensor, [0, 2, 1, 3, 4])

        reshaped_input_tensor = tf.reshape(perm_output_tensor, shape=(in_b * in_w, in_h,
                                                                      in_t, filters))
        conv2d_layer_name_2 = name_prefix + "Conv2D_{}_2".format(filters)
        conv2d_2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                          name=conv2d_layer_name_2, activation=activation,
                                          kernel_regularizer=self.kernel_regularizer, data_format='channels_last')
        setattr(self, conv2d_layer_name_2, conv2d_2)

        conv2d_2 = conv2d_2(reshaped_input_tensor)
        conv2d_2 = self._add_regularization_layer(conv2d_2, name_suffix=conv2d_layer_name_2)

        output = tf.reshape(conv2d_2, shape=(in_b, in_w, in_h, in_t, filters))

        return output

    def _get_convolution_transpose_layer(self, input_layer, filters, kernel_size=3, strides=(2, 2, 1), padding='same',
                                         name_prefix='r_', activation=tf.keras.activations.relu):

        conv3d_transpose_layer_name = name_prefix + "UpConv3D_{}".format(filters)
        conv3d_transpose = tf.keras.layers.Convolution3DTranspose(filters=filters, kernel_size=kernel_size,
                                                                  strides=strides, padding=padding,
                                                                  activation=activation,
                                                                  name=conv3d_transpose_layer_name,
                                                                  kernel_regularizer=self.kernel_regularizer)

        setattr(self, conv3d_transpose_layer_name, conv3d_transpose)
        conv3d_transpose = conv3d_transpose(input_layer)
        conv3d_transpose = self._add_regularization_layer(conv3d_transpose, name_suffix=conv3d_transpose_layer_name,
                                                          input_type='3d')
        return conv3d_transpose

    def _get_max_pool_3d(self, filters, pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same',
                         name_prefix='l_'):

        maxpool_3d_layer_name = name_prefix + "MaxPool3D_{}".format(filters)
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
                current_layer = self._get_max_pool_3d(filters=filters)(first_layer)
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
        conv3d_1_1 = self._add_regularization_layer(conv3d_1_1, name_suffix='conv3d_1_1', input_type='3d')

        sum = tf.keras.layers.add([conv3d_1_1, inputs])
        update = tf.keras.activations.relu(sum)
        output = tf.keras.backend.squeeze(update, -1)
        return output
