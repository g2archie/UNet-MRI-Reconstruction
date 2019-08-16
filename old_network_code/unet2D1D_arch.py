# ==================================================================================================
# The procedures encapsulate architecture of 3 dimensional UNet. Convolution layers treat spatial
# and temporal dimensions separately, resulting in performing 2D convolutions on the former and
# 1D convolutions on the latter.
# ==================================================================================================
# --------------------------------------------------------------------------------------------------

import tensorflow as tf

# --------------------------------------------------------------------------------------------------
def conv3DTranspose\
(
    in_tensor,
    filter,
    shape
):
    """
    Performs up convolution constant in temporal domain i.e. in_tensor.shape[3] parameter.

    :param in_tensor: Input tensor of shape: [ batch, img_width, img_height, time, in_channels ]
    :param filter:    Tensor of shape: [ width, height, depth, out_channels, in_channels ]
    :param shape:     A 1-D Tensor representing the output shape of the deconvolution op.
    """
    return tf.nn.conv3d_transpose\
    (
        in_tensor,
        filter,
        shape,
        strides = [1, 2, 2, 1, 1],
        padding = 'SAME'
    )


# --------------------------------------------------------------------------------------------------
def maxPool2x2x1\
(
    in_tensor
):
    """
    Downsamples provided tensor by 2x in spatial domain, leaving temporal domain unchanged.

    :param in_tensor: Input tensor of shape: [ batch, img_width, img_height, time, in_channels ]
    """

    return tf.nn.max_pool3d\
    (
        in_tensor,
        ksize   = [1, 2, 2, 1, 1],
        strides = [1, 2, 2, 1, 1],
        padding = 'SAME'
    )


# --------------------------------------------------------------------------------------------------
def constantTensor\
(
    shape,
    is_variable = True,
    value = 0.025
):
    """
    Creates a constant tensor either as a tf.Variable or not.

    :param shape:       dimensions of resulting tensor.
    :param is_variable: if true, created tensor will be added to the graph. Otherwise not.
    :param value:       a constant value (or list) of output type dtype.
    """
    constant_tensor = tf.constant( value, shape = shape )
    if is_variable:
        return tf.Variable( constant_tensor )
    else:
        return constant_tensor

# --------------------------------------------------------------------------------------------------
def randomTensor\
(
    shape,
    is_variable = True,
    standard_deviation = 0.025
):
    """
    Outputs a tensor filled with random values from a truncated normal distribution.
    The generated values follow a normal distribution with specified mean and standard deviation,
    except that values whose magnitude is more than 2 standard deviations from the mean are
    dropped and re-picked.

    :param shape:              a 1-D integer Tensor or Python array. The shape of the output tensor
    :param is_variable:        if true, created tensor will be added to the graph. Otherwise not.
    :param standard_deviation: a 0-D Tensor or Python value of type dtype. The standard deviation
                               of the normal distribution, before truncation.
    """
    randomized_tensor = tf.truncated_normal( shape, stddev = standard_deviation )
    if is_variable:
        return tf.Variable( randomized_tensor )
    else:
        return randomized_tensor


# --------------------------------------------------------------------------------------------------
def upConv3DLayer2x2x1\
(
    in_tensor,
    out_channels
):
    """
    Applies up-convolution on input tensor followed by ReLU. The process will increase img_width
    and img_height of the tensor by 2.

    :param in_tensor:     A TF's tensor that one wants to up convolve.
                          It is assumed that the the tensor is of following form:
                          [batch, img_width, img_height, time, channels]
    :param out_channels:  Number of channels in the output tensor.
    """

    # Extract input tensor's size: [batch, img_width, img_height, time, channels]
    in_b, in_w, in_h, in_t, in_c = in_tensor.get_shape().as_list()

    # Output layer shape
    out_layer_w = int( in_w * 2 )
    out_layer_h = int( in_h * 2 )
    out_layer_t = int( in_t     )

    # Up convolution and ReLU parameters
    filter        = randomTensor( [3, 3, 3, out_channels, in_c] )
    up_conv_shape = [ in_b, out_layer_w, out_layer_h, out_layer_t, out_channels ]
    bias          = constantTensor( [out_channels] )

    return tf.nn.relu\
    (
        conv3DTranspose( in_tensor, filter, up_conv_shape )
        + bias
    )

# --------------------------------------------------------------------------------------------------
def conv2D1D\
(
    in_tensor,
    out_channels
):
    """
    The procedure applies conv2D1D to the input tensor. The convolution will use tf.nn.conv2d over
    spatial data and tf.nn.conv1d over temporal domain. This way it learns about per-image pixels
    correlations and also take into account per-pixel change in time.

    :param in_tensor:     A TF's tensor that one wants to convolve.
                          It is assumed that the the tensor is of the following form:
                          [ batch, img_width, img_height, time, channels ]
    :param out_channels:  Number of channels in the output tensor.
    """

    # Extract input tensor's size: [batch, img_width, img_height, time, channels]
    in_b, in_w, in_h, in_t, in_c = in_tensor.get_shape().as_list()

    # Create one and two dimensional filters and add them to the graph.
    filter_tensor_2D = randomTensor( [ 3, 3, in_c, out_channels ] )
    filter_tensor_1D = randomTensor( [ 3, out_channels, out_channels          ] )

    # 2D convolution over spatial slices:
    # step_a:   transpose from [in_b, in_w, in_h, in_t, in_c] to [ in_b, in_t, in_w, in_h, in_c ]
    # step_b:   merge time slices into batches producing [ in_b * in_t, in_w, in_h, in_c ] tensor
    # step_c:   convolve and output [ in_b * in_t, in_w, in_h, out_channels ] tensor
    # step_d:   reshape it back to the original number of batches and time slices.
    # final_2D: transpose, so time slices are back i.e. [ in_b, in_w, in_h, in_t, out_channels ]
    step_a   = tf.transpose ( in_tensor, [ 0, 3, 1, 2, 4 ]                                )
    step_b   = tf.reshape   ( step_a, shape = ( in_b * in_t, in_w, in_h, in_c )           )
    step_c   = tf.nn.conv2d ( step_b, filter_tensor_2D, [ 1, 1, 1, 1 ], "SAME"            )
    step_d   = tf.reshape   ( step_c, shape = ( in_b, in_t, in_w, in_h, out_channels )    )
    final_2D = tf.transpose ( step_d, [ 0, 2, 3, 1, 4 ]                                   )

    # 1D convolution over time:
    # step_e:   merge spatial domain into batches. This should result in number of batches equal
    #           to original_batch_size * number_of_pixels_per_image. Each batch then encapsulates
    #           a pixel's change in time in different filters.
    # step_f:   convolve over time domain. Number of filters remain the same.
    # final_1D: Restore original size of the tensor.
    step_e   = tf.reshape   ( final_2D, shape = ( in_b *in_w * in_h, in_t, out_channels ) )
    step_f   = tf.nn.conv1d ( step_e, filter_tensor_1D, 1, "SAME"                         )
    final_1D = tf.reshape   ( step_f,   shape = ( in_b, in_w, in_h, in_t, out_channels )  )

    return final_1D

# --------------------------------------------------------------------------------------------------
def doubleConv2D1DLayer\
(
    in_tensor,
    out_channels
):
    """
    The procedure applies (conv2D1D + ReLU) twice to the input tensor.

    :param in_tensor:     A TF's tensor that one wants to double-convolve.
                          It is assumed that the the tensor is of the following form:
                          [ batch, img_width, img_height, time, channels ]
    :param out_channels:  Number of channels in the output tensor.
    """

    # First convolutional layer - maps input channels to 'out_channels' filters
    # using conv2D over spatial data and conv1D over temporal data.
    b_1 = constantTensor ( [out_channels]                            )
    h_1 = tf.nn.relu     ( conv2D1D( in_tensor, out_channels ) + b_1 )

    # Second convolutional layer - maps 'out_channels' filters to 'out_channels' filters
    # using conv2D over spatial data and conv1D over temporal data.
    b_2 = constantTensor ( [out_channels]                            )
    h_2 = tf.nn.relu     ( conv2D1D( h_1, out_channels ) + b_2       )

    return h_2

# --------------------------------------------------------------------------------------------------
def unet2D1DArchitecture\
(
    in_image
):
    """
    Constructs UNet that is capable of processing 3D data. Its original double-convolution layers
    were replaced with something called conv2D1D (please refer to the procedure itself for more
    information). On top of that, the network up and down samples only in spatial domain leaving
    temporal data untouched in terms of its dimensionality

    :param in_image: Tensor of input image
    """
    x_image = tf.expand_dims( in_image, -1 )

    # Downsampling
    h_dconv32f  = doubleConv2D1DLayer    ( x_image,       32                  )
    h_maxpool1  = maxPool2x2x1           ( h_dconv32f                         )
    h_dconv64f  = doubleConv2D1DLayer    ( h_maxpool1,    64                  )
    h_maxpool2  = maxPool2x2x1           ( h_dconv64f                         )
    h_dconv128f = doubleConv2D1DLayer    ( h_maxpool2,    128                 )
    h_maxpool3  = maxPool2x2x1           ( h_dconv128f                        )
    h_dconv256f = doubleConv2D1DLayer    ( h_maxpool3,    256                 )
    h_maxpool4  = maxPool2x2x1           ( h_dconv256f                        )
    h_dconv512f = doubleConv2D1DLayer    ( h_maxpool4,    512                 )

    # Upsampling
    h2_updconv256f = upConv3DLayer2x2x1  ( h_dconv512f,   256                 )
    h2_concat512f  = tf.concat           ( [ h_dconv256f, h2_updconv256f ], 4 )
    h2_dconv256f   = doubleConv2D1DLayer ( h2_concat512f, 256                 )
    h2_updconv128f = upConv3DLayer2x2x1  ( h2_dconv256f,  128                 )
    h2_concat256f  = tf.concat           ( [ h_dconv128f, h2_updconv128f ], 4 )
    h2_dconv128f   = doubleConv2D1DLayer ( h2_concat256f, 128                 )
    h2_updconv64f  = upConv3DLayer2x2x1  ( h2_dconv128f,  64                  )
    h2_concat128f  = tf.concat           ( [ h_dconv64f,  h2_updconv64f ],  4 )
    h2_dconv64f    = doubleConv2D1DLayer ( h2_concat128f, 64                  )
    h2_updconv32f  = upConv3DLayer2x2x1  ( h2_dconv64f,   32                  )
    h2_concat64f   = tf.concat           ( [ h_dconv32f,  h2_updconv32f ],  4 )
    h2_dconv32f    = doubleConv2D1DLayer ( h2_concat64f,  32                  )

    # Last convolution - no ReLU
    b_convDownB1_3 = constantTensor( [1] )
    h_convDownB1_3 = conv2D1D( h2_dconv32f, 1 ) + b_convDownB1_3

    h_sum = tf.add( h_convDownB1_3, x_image )

    h_update = tf.nn.relu( h_sum )
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

    imag = tf.placeholder( tf.float32, [ batchSize, inputShape[1], inputShape[2], inputShape[3] ] )
    true = tf.placeholder( tf.float32, [ batchSize, inputShape[1], inputShape[2], inputShape[3] ] )

    y_conv = unet2D1DArchitecture( imag )

    # Define loss and optimizer
    loss = tf.nn.l2_loss( true - y_conv )
    added_loss = -tf.scalar_mul\
    (
        100.0,
        tf.minimum
        (
            tf.subtract( tf.norm( y_conv ), 1.0 ),
            0.0
        )
    )

    train_step = tf.train.AdamOptimizer(1e-3).minimize( tf.add( loss, added_loss ) )
    accuracy   = tf.reduce_mean( tf.cast( loss, tf.float32 ) )
    testPos    = tf.reduce_mean( tf.cast( added_loss, tf.float32 ) )
    saver      = tf.train.Saver()

    return NetworkArchitectureWrapper\
    (
        y_conv = y_conv,
        train_step = train_step,
        accuracy = accuracy,
        test_pos = testPos,
        saver = saver,
        imag = imag,
        true = true
    )


