import tensorflow as tf

from layers.instancenormalization import InstanceNormalization
from layers.dropblock import DropBlock3D


class UNet2D1D(tf.keras.Model):

    def __init__(self, regularization=None, regularization_parameters=None):
        super().__init__()
        self.regularization = regularization
        self.regularization_parameters = regularization_parameters
        self.no_of_conv_layers = 23
        self.kernel_regularizer = None
        self.regularization_layers = []
        if regularization is not None:
            if regularization == 'l2':
                self.kernel_regularizer = tf.keras.regularizers.l2(*regularization_parameters)
            elif regularization == 'l1':
                self.kernel_regularizer = tf.keras.regularizers.l1(*regularization_parameters)
            elif regularization == 'l1_l2':
                self.kernel_regularizer = tf.keras.regularizers.l1_l2(*regularization_parameters)

        self.conv2d_32_1 = tf.keras.layers.Conv3D(filters=32, kernel_size=3, strides=1, padding='same',
                                                  activation=tf.nn.relu, name='conv3d_32_1',
                                                  kernel_regularizer=self.kernel_regularizer)
        self.conv1d_32_2 = tf.keras.layers.Conv3D(filters=32, kernel_size=3, strides=1, padding='same',
                                                  activation=tf.nn.relu, name='conv3d_32_2',
                                                  kernel_regularizer=self.kernel_regularizer)
        self.maxpool_3d_1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same',
                                                         name='maxpool_3d_1')
        self.conv2d_64_1 = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=1, padding='same',
                                                  activation=tf.nn.relu, name='conv3d_64_1',
                                                  kernel_regularizer=self.kernel_regularizer)
        self.conv3d_64_2 = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=1, padding='same',
                                                  activation=tf.nn.relu, name='conv3d_64_2',
                                                  kernel_regularizer=self.kernel_regularizer)
        self.maxpool_3d_2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same',
                                                         name='maxpool_3d_2')
        self.conv3d_128_1 = tf.keras.layers.Conv3D(filters=128, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.relu, name='conv3d_128_1',
                                                   kernel_regularizer=self.kernel_regularizer)

        self.conv3d_128_2 = tf.keras.layers.Conv3D(filters=128, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.relu, name='conv3d_128_2',
                                                   kernel_regularizer=self.kernel_regularizer)
        self.maxpool_3d_3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same',
                                                         name='maxpool_3d_3')
        self.conv3d_256_1 = tf.keras.layers.Conv3D(filters=256, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.relu, name='conv3d_256_1',
                                                   kernel_regularizer=self.kernel_regularizer)
        self.conv3d_256_2 = tf.keras.layers.Conv3D(filters=256, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.relu, name='conv3d_256_2',
                                                   kernel_regularizer=self.kernel_regularizer)
        self.maxpool_3d_4 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same',
                                                         name='maxpool_3d_4')
        self.conv3d_512_1 = tf.keras.layers.Conv3D(filters=512, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.relu, name='conv3d_512_1',
                                                   kernel_regularizer=self.kernel_regularizer)
        self.conv3d_512_2 = tf.keras.layers.Conv3D(filters=512, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.relu, name='conv3d_512_2',
                                                   kernel_regularizer=self.kernel_regularizer)

        self.conv3d_trans_256_1 = tf.keras.layers.Convolution3DTranspose(filters=256, kernel_size=3, strides=(2, 2, 1),
                                                                         padding='same', activation=tf.nn.relu,
                                                                         name='conv3d_trans_256_1',
                                                                         kernel_regularizer=self.kernel_regularizer)
        self.concat_256_1 = tf.keras.layers.Concatenate(axis=-1, name='concat_256_1')
        self.conv3d_256_3 = tf.keras.layers.Conv3D(filters=256, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.relu, name='conv3d_256_3',
                                                   kernel_regularizer=self.kernel_regularizer)
        self.conv3d_256_4 = tf.keras.layers.Conv3D(filters=256, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.relu, name='conv3d_256_4',
                                                   kernel_regularizer=self.kernel_regularizer)
        self.conv3d_trans_128_1 = tf.keras.layers.Convolution3DTranspose(filters=128, kernel_size=3, strides=(2, 2, 1),
                                                                         padding='same', activation=tf.nn.relu,
                                                                         name='conv3d_trans_128_1',
                                                                         kernel_regularizer=self.kernel_regularizer)
        self.concat_128_1 = tf.keras.layers.Concatenate(axis=-1, name='concat_128_1')
        self.conv3d_128_3 = tf.keras.layers.Conv3D(filters=128, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.relu, name='conv3d_128_3',
                                                   kernel_regularizer=self.kernel_regularizer)
        self.conv3d_128_4 = tf.keras.layers.Conv3D(filters=128, kernel_size=3, strides=1, padding='same',
                                                   activation=tf.nn.relu, name='conv3d_128_4',
                                                   kernel_regularizer=self.kernel_regularizer)
        self.conv3d_trans_64_1 = tf.keras.layers.Convolution3DTranspose(filters=64, kernel_size=3, strides=(2, 2, 1),
                                                                        padding='same', activation=tf.nn.relu,
                                                                        name='conv3d_trans_64_1',
                                                                        kernel_regularizer=self.kernel_regularizer)
        self.concat_64_1 = tf.keras.layers.Concatenate(axis=-1, name='concat_64_1')
        self.conv3d_64_3 = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=1, padding='same',
                                                  activation=tf.nn.relu, name='conv3d_64_3',
                                                  kernel_regularizer=self.kernel_regularizer)
        self.conv3d_64_4 = tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=1, padding='same',
                                                  activation=tf.nn.relu, name='conv3d_64_4',
                                                  kernel_regularizer=self.kernel_regularizer)

        self.conv3d_trans_32_1 = tf.keras.layers.Convolution3DTranspose(filters=32, kernel_size=3, strides=(2, 2, 1),
                                                                        padding='same', name='conv3d_trans_32_1',
                                                                        kernel_regularizer=self.kernel_regularizer)
        self.concat_32_1 = tf.keras.layers.Concatenate(axis=-1, name='concat_32_1')
        self.conv3d_32_3 = tf.keras.layers.Conv3D(filters=32, kernel_size=3, strides=1, padding='same',
                                                  activation=tf.nn.relu, name='conv3d_32_3',
                                                  kernel_regularizer=self.kernel_regularizer)
        self.conv3d_32_4 = tf.keras.layers.Conv3D(filters=32, kernel_size=3, strides=1, padding='same',
                                                  activation=tf.nn.relu, name='conv3d_32_4',
                                                  kernel_regularizer=self.kernel_regularizer)

        self.conv3d_1_1 = tf.keras.layers.Conv3D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3d_1_1',
                                                 kernel_regularizer=self.kernel_regularizer)

    def _add_regularization_layer(self, input_layer, name_prefix, filters):

        if self.regularization == 'batch_norm':
            return tf.keras.layers.BatchNormalization(-1, name=name_prefix + "Batch_Norm_{}".format(filters))(
                input_layer)

        elif self.regularization == 'instance_norm':
            return InstanceNormalization(-1, name=name_prefix + "Instance_Norm_{}".format(filters))(input_layer)

        elif self.regularization == 'dropout':
            return tf.keras.layers.Dropout(*self.regularization_parameters,
                                           name=name_prefix + 'Dropout_{}'.format(filters))(input_layer)
        elif self.regularization == 'dropblock':
            return DropBlock3D(*self.regularization_parameters, name=name_prefix + 'DropBlock_{}'.format(filters))(
                input_layer)

    def _get_convolution_block(self, input_layer, filters, kernel_size=3, strides=1, padding='same',
                               name_prefix='left_', activation=tf.keras.activations.relu):

        input_layer_shape = tf.shape(input_layer)

        perm_input_tensor = tf.transpose(input_layer, perm=[0, 3, 1, 2, 4])
        reshaped_input_tensor = tf.reshape(perm_input_tensor, shape=(input_layer_shape[0] * input_layer_shape[3],
                                                                     input_layer_shape[1], input_layer_shape[2],
                                                                     input_layer_shape[4]))
        conv2d_layer_name = name_prefix + "Conv2D_{}".format(filters)
        conv2d = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                        name=conv2d_layer_name, activation=activation,
                                        kernel_regularizer=self.kernel_regularizer)
        setattr(self, conv2d_layer_name, conv2d)

        conv2d = conv2d(reshaped_input_tensor)
        conv2d = self._add_regularization_layer(conv2d, name_prefix=name_prefix, filters=filters)

        reshaped_output_tensor = tf.reshape(conv2d, shape=(input_layer_shape[0], input_layer_shape[3],
                                                           input_layer_shape[1], input_layer_shape[2], filters))

        perm_output_tensor = tf.transpose(reshaped_output_tensor, [0, 2, 3, 1, 4])

        reshaped_input_tensor = tf.reshape(perm_output_tensor, shape=(input_layer[0] * input_layer[1] * input_layer[2],
                                                                      input_layer[3], input_layer[4]))
        conv1d_layer_name = name_prefix + "Conv1D_{}".format(filters)
        conv1d = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                        name=conv1d_layer_name, activation=activation,
                                        kernel_regularizer=self.kernel_regularizer)
        setattr(self, conv1d_layer_name, conv1d)

        conv1d = conv1d(reshaped_input_tensor)
        conv1d = self._add_regularization_layer(conv1d, name_prefix=name_prefix, filters=filters)

        output = tf.reshape(conv1d, shape=input_layer_shape)
        return output

    def _get_convolution_transpose_layer(self, filters, kernel_size=3, strides=(2, 2, 1), padding='same',
                                         name_prefix='right_', activation=tf.keras.activations.relu):

        conv3d_transpose_layer_name = name_prefix + "UpConv3D_{}".format(filters)
        conv3d_transpose = tf.keras.layers.Convolution3DTranspose(filters=filters, kernel_size=kernel_size,
                                                                  strides=strides, padding=padding,
                                                                  activation=activation,
                                                                  name=conv3d_transpose_layer_name,
                                                                  kernel_regularizer=self.kernel_regularizer)
        setattr(self, conv3d_transpose_layer_name, conv3d_transpose)

        return conv3d_transpose

    def call(self, inputs):

        inputs = tf.keras.backend.expand_dims(inputs, axis=-1)

        conv3d_32_1 = self.conv2d_32_1(inputs)
        if self.regularization_layers:
            conv3d_32_1 = self.regularization_layers[0](conv3d_32_1)

        conv3d_32_2 = self.conv1d_32_2(conv3d_32_1)
        if self.regularization_layers:
            conv3d_32_2 = self.regularization_layers[1](conv3d_32_2)

        maxpool_3d_1 = self.maxpool_3d_1(conv3d_32_2)

        conv3d_64_1 = self.conv2d_64_1(maxpool_3d_1)
        if self.regularization_layers:
            conv3d_64_1 = self.regularization_layers[2](conv3d_64_1)

        conv3d_64_2 = self.conv3d_64_2(conv3d_64_1)
        if self.regularization_layers:
            conv3d_64_2 = self.regularization_layers[3](conv3d_64_2)

        maxpool_3d_2 = self.maxpool_3d_2(conv3d_64_2)

        conv3d_128_1 = self.conv3d_128_1(maxpool_3d_2)
        if self.regularization_layers:
            conv3d_128_1 = self.regularization_layers[4](conv3d_128_1)

        conv3d_128_2 = self.conv3d_128_2(conv3d_128_1)
        if self.regularization_layers:
            conv3d_128_2 = self.regularization_layers[5](conv3d_128_2)

        maxpool_3d_3 = self.maxpool_3d_3(conv3d_128_2)

        conv3d_256_1 = self.conv3d_256_1(maxpool_3d_3)
        if self.regularization_layers:
            conv3d_256_1 = self.regularization_layers[6](conv3d_256_1)

        conv3d_256_2 = self.conv3d_256_2(conv3d_256_1)
        if self.regularization_layers:
            conv3d_256_2 = self.regularization_layers[7](conv3d_256_2)

        maxpool_3d_4 = self.maxpool_3d_4(conv3d_256_2)

        conv3d_512_1 = self.conv3d_512_1(maxpool_3d_4)
        if self.regularization_layers:
            conv3d_512_1 = self.regularization_layers[8](conv3d_512_1)

        conv3d_512_2 = self.conv3d_512_2(conv3d_512_1)
        if self.regularization_layers:
            conv3d_512_2 = self.regularization_layers[9](conv3d_512_2)

        conv3d_trans_256_1 = self.conv3d_trans_256_1(conv3d_512_2)
        if self.regularization_layers:
            conv3d_trans_256_1 = self.regularization_layers[10](conv3d_trans_256_1)

        concat_256_1 = self.concat_256_1([conv3d_256_2, conv3d_trans_256_1])

        conv3d_256_3 = self.conv3d_256_3(concat_256_1)
        if self.regularization_layers:
            conv3d_256_3 = self.regularization_layers[11](conv3d_256_3)

        conv3d_256_4 = self.conv3d_256_4(conv3d_256_3)
        if self.regularization_layers:
            conv3d_256_4 = self.regularization_layers[12](conv3d_256_4)

        conv3d_trans_128_1 = self.conv3d_trans_128_1(conv3d_256_4)
        if self.regularization_layers:
            conv3d_trans_128_1 = self.regularization_layers[13](conv3d_trans_128_1)
        # concat_128_1 = tf.keras.layers.concatenate([conv3d_128_2, conv3d_trans_128_1], axis=-1)
        concat_128_1 = self.concat_128_1([conv3d_128_2, conv3d_trans_128_1])

        conv3d_128_3 = self.conv3d_128_3(concat_128_1)
        if self.regularization_layers:
            conv3d_128_3 = self.regularization_layers[14](conv3d_128_3)

        conv3d_128_4 = self.conv3d_128_4(conv3d_128_3)
        if self.regularization_layers:
            conv3d_128_4 = self.regularization_layers[15](conv3d_128_4)

        conv3d_trans_64_1 = self.conv3d_trans_64_1(conv3d_128_4)
        if self.regularization_layers:
            conv3d_trans_64_1 = self.regularization_layers[16](conv3d_trans_64_1)

        concat_64_1 = self.concat_64_1([conv3d_64_2, conv3d_trans_64_1])

        conv3d_64_3 = self.conv3d_64_3(concat_64_1)
        if self.regularization_layers:
            conv3d_64_3 = self.regularization_layers[17](conv3d_64_3)

        conv3d_64_4 = self.conv3d_64_4(conv3d_64_3)
        if self.regularization_layers:
            conv3d_64_4 = self.regularization_layers[18](conv3d_64_4)

        conv3d_trans_32_1 = self.conv3d_trans_32_1(conv3d_64_4)
        if self.regularization_layers:
            conv3d_trans_32_1 = self.regularization_layers[19](conv3d_trans_32_1)

        concat_32_1 = self.concat_32_1([conv3d_32_2, conv3d_trans_32_1])

        conv3d_32_3 = self.conv3d_32_3(concat_32_1)
        if self.regularization_layers:
            conv3d_32_3 = self.regularization_layers[20](conv3d_32_3)

        conv3d_32_4 = self.conv3d_32_4(conv3d_32_3)
        if self.regularization_layers:
            conv3d_32_4 = self.regularization_layers[21](conv3d_32_4)

        conv3d_1_1 = self.conv3d_1_1(conv3d_32_4)
        if self.regularization_layers:
            conv3d_1_1 = self.regularization_layers[22](conv3d_1_1)

        sum = tf.keras.layers.add([conv3d_1_1, inputs])

        update = tf.keras.activations.relu(sum)
        output = tf.keras.backend.squeeze(update, -1)

        return output
