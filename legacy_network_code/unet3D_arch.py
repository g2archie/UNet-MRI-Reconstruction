import tensorflow as tf

def doubleConv3DLayer(inTensor, inChannels, outChannels):
    # TODO: change naming
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_convPrioB1_1 = weight_variable([3, 3, 3, inChannels, outChannels])
    b_convPrioB1_1 = bias_variable([outChannels])
    h_convPrioB1_1 = tf.nn.relu(conv3d(inTensor, W_convPrioB1_1) + b_convPrioB1_1)

    # Second convolutional layer -- maps 32 feature maps to 32.
    W_convPrioB1_2 = weight_variable([3, 3, 3, outChannels, outChannels])
    b_convPrioB1_2 = bias_variable([outChannels])

    return tf.nn.relu(conv3d(h_convPrioB1_1, W_convPrioB1_2) + b_convPrioB1_2)

def upConv3DLayer(inTensor, inChannels, outChannels, bSize):
    layerSizeX = int( inTensor.shape[1] * 2 )
    layerSizeY = int( inTensor.shape[2] * 2 )
    layerSizeZ = int( inTensor.shape[3] )

    W_TconvPrioB3 = weight_variable([3, 3, 3, outChannels, inChannels])  # Ouput, Input channels
    b_TconvPrioB3 = bias_variable([outChannels])
    return tf.nn.relu( conv3d_trans(inTensor, W_TconvPrioB3, [int(bSize), layerSizeX, layerSizeY, layerSizeZ, outChannels]) + b_TconvPrioB3 )

def conv3d\
(
    x,
    W
):
    """
    conv3d returns a 3d convolution layer with full stride.
    :param x:
    :param W:
    :return:
    """
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def conv3d_trans(x, W, shape):
    """conv3d returns a 3d convolution layer with full stride."""
    return tf.nn.conv3d_transpose(x, W, shape, strides=[1, 2, 2, 1, 1], padding='SAME')


def max_pool_2x2x1(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 1, 1],
                            strides=[1, 2, 2, 1, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.025)
    return tf.Variable(initial, trainable=True)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.025, shape=shape)
    return tf.Variable(initial, trainable=True)


def step_length():
    """Step length before update for a given shape."""
    initial = tf.constant(1.0)
    return tf.Variable(initial)

def unet3DArchitecture\
(
    in_image
):
    """ Constructs 3D version of UNet - downsampling happens only in spatial domain."""
    # x_image = tf.reshape( x, [-1, imSize[1], imSize[2], imSize[3], 1] )
    x_image = tf.expand_dims(in_image, -1)
    #x_image = in_image
    # Downsampling
    h_dconv32f  = doubleConv3DLayer ( x_image,    1,   32  )
    h_maxpool1  = max_pool_2x2x1    ( h_dconv32f           )
    h_dconv64f  = doubleConv3DLayer ( h_maxpool1, 32,  64  )
    h_maxpool2  = max_pool_2x2x1    ( h_dconv64f           )
    h_dconv128f = doubleConv3DLayer ( h_maxpool2, 64,  128 )
    h_maxpool3  = max_pool_2x2x1    ( h_dconv128f          )
    h_dconv256f = doubleConv3DLayer ( h_maxpool3, 128, 256 )
    h_maxpool4  = max_pool_2x2x1    ( h_dconv256f          )
    h_dconv512f = doubleConv3DLayer ( h_maxpool4, 256, 512 )

    # Upsampling TODO: consider moving batch to optional parameter
    h2_updconv256f = upConv3DLayer     ( h_dconv512f,   512, 256, 8         )
    h2_concat512f  = tf.concat         ( [ h_dconv256f, h2_updconv256f ], 4 )
    h2_dconv256f   = doubleConv3DLayer ( h2_concat512f, 512, 256            )

    h2_updconv128f = upConv3DLayer     ( h2_dconv256f,  256, 128, 8         )
    h2_concat256f  = tf.concat         ( [ h_dconv128f, h2_updconv128f ], 4 )
    h2_dconv128f   = doubleConv3DLayer ( h2_concat256f, 256, 128            )
    h2_updconv64f  = upConv3DLayer     ( h2_dconv128f,  128, 64,  8         )
    h2_concat128f  = tf.concat         ( [ h_dconv64f, h2_updconv64f ], 4   )
    h2_dconv64f    = doubleConv3DLayer ( h2_concat128f, 128, 64             )
    h2_updconv32f  = upConv3DLayer     ( h2_dconv64f,   64,  32,  8         )
    h2_concat64f   = tf.concat         ( [ h_dconv32f, h2_updconv32f ], 4   )
    h2_dconv32f    = doubleConv3DLayer ( h2_concat64f,  64,  32             )

    # Conv Down - no RELU!
    W_convDownB1_3 = weight_variable([3, 3, 3, 32, 1])
    b_convDownB1_3 = bias_variable([1])
    h_convDownB1_3 = conv3d(h2_dconv32f, W_convDownB1_3) + b_convDownB1_3

    h_sum = tf.add(h_convDownB1_3, x_image)

    h_update = tf.nn.relu(h_sum)
    h_update = tf.clip_by_value(h_update, 0, 1)
    h_update = tf.squeeze( h_update )

    return h_update

# --------------------------------------------------------------------------------------------------
class NetworkArchitectureWrapper:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr( self, k, v )

# --------------------------------------------------------------------------------------------------
def create\
(
    inputShape,
    batchSize
):
    regularization_parameter = 1e-3
    imag = tf.placeholder(tf.float32, [batchSize, inputShape[1], inputShape[2], inputShape[3]])
    true = tf.placeholder(tf.float32, [batchSize, inputShape[1], inputShape[2], inputShape[3]])
    y_conv = unet3DArchitecture( imag )
    print("shape of true is {}, shape of y_conv is {}".format(tf.shape(true), tf.shape(y_conv)))    
    # Define loss and optimizer
    #loss = tf.reduce_mean(tf.abs(true-y_conv))
    #loss = tf.nn.l2_loss(true - y_conv)
    #loss = _ssim_loss(true, y_conv, batchSize)
    #loss = tf.reduce_sum(tf.square(true-y_conv)) 
    #loss_2 = tf.reduce_sum(tf.abs(true-y_conv))
    loss = tf.reduce_mean(tf.subtract(1.0, tf.image.ssim(y_conv, true, 1.0)))
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=true, logits=y_conv))
    #added_loss = -tf.scalar_mul(100.0, tf.minimum(tf.subtract(tf.norm(y_conv), 1.0), 0.0))
    
    #added_loss = tf.constant(0.0)
    added_loss = tf.reduce_mean(tf.scalar_mul(regularization_parameter, tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]))) #l2 loss
    #print("shape of loss is {}, shape loss_2 is {}, shape of added_loss is {}, shape of l2 regularization is {}".format(tf.shape(true), tf.shape(y_conv), tf.shape(added_loss), tf.shape(added_loss_2)))  
    #added_loss_length = len([v.name for v in tf.trainable_variables()])
    #print("regularization term length is {}".format(added_loss_length))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(tf.add(loss, added_loss))
    #accuracy = tf.cast(loss, tf.float32)
    accuracy   = tf.reduce_mean(tf.cast(loss, tf.float32))
    #accuracy = tf.reduce_mean(loss)
    testPos    = tf.reduce_mean(tf.cast(added_loss, tf.float32))
    saver      = tf.train.Saver()

    return NetworkArchitectureWrapper\
    (
        y_conv = y_conv,
        train_step = train_step,
        accuracy = accuracy,
        test_pos = testPos,
        saver = saver,
        imag = imag,
        true = true,
        added_loss = added_loss,
        loss = loss
    )

    
    

