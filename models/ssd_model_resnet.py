import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

kernel_initializer = 'he_normal' 
# kernel_regularizer = l2(1.e-4)

def _shortcut(x, residual, block_name, dropout=0.0):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    block_name = block_name+'/shortcut'
    with K.name_scope(block_name) as scope:
        input_shape = K.int_shape(x)
        residual_shape = K.int_shape(residual)
        stride_width = int(round(input_shape[1] / residual_shape[1]))
        stride_height = int(round(input_shape[2] / residual_shape[2]))
        equal_channels = input_shape[3] == residual_shape[3]

        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            shortcut = Conv2D(filters=residual_shape[3], kernel_size=(1,1),
                              strides=(stride_width, stride_height), padding="same", 
                              kernel_initializer=kernel_initializer, name=block_name+'/conv')(x)

            shortcut = Dropout(dropout, name=block_name+'/dropout')(shortcut)
        else:
            shortcut = x
        return add([shortcut, residual], name=block_name+'/add')

def _bn_relu_conv(x, filters, kernel_size, block_name, dropout=0.0, strides=(1,1), padding="same"):
    with K.name_scope(block_name) as scope:
        x = BatchNormalization(axis=3)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, kernel_size, strides=strides, padding=padding, 
                   kernel_initializer=kernel_initializer, name=block_name+'/conv')(x)

        x = Dropout(dropout, name=block_name+'/dropout')(x)
    return x

def bl_bottleneck(x, filters, block_name, dropout=0.0, add_bottleneck_blk=False, kernel_size=(3, 3), strides=(1,1), is_first_layer_of_first_block=False):
    if is_first_layer_of_first_block:
        # don't repeat bn->relu since we just did bn->relu->maxpool
        x1 = Conv2D(filters, (1,1), strides=strides, padding="same", 
                    kernel_initializer=kernel_initializer, name=block_name+'/bn_relu_conv_1')(x)
    else:
        x1 = _bn_relu_conv(x, filters=filters, dropout=dropout, kernel_size=(1,1), strides=strides, block_name=block_name+'/bn_relu_conv_1')
    x1 = _bn_relu_conv(x1, filters=filters, dropout=dropout, kernel_size=kernel_size, block_name=block_name+'/bn_relu_conv_2')
    if add_bottleneck_blk:
        x1 = _bn_relu_conv(x1, filters=filters*4, dropout=dropout, kernel_size=(1,1), block_name=block_name+'/bn_relu_conv_3')
    x1 = Dropout(dropout, name=block_name+'/dropout')(x1)
    return _shortcut(x, x1, block_name, dropout=dropout)


def ssd512_resnet_body(x, activation='relu',dropout=0.0):
    
    if activation == 'leaky_relu':
        activation = leaky_relu
    
    source_layers = []
    
    block_name = 'block_1'
    with K.name_scope(block_name) as scope:
        x = Conv2D(32, (7,7), strides=(2,2), padding='same', 
                   kernel_initializer=kernel_initializer, name=block_name+'/conv')(x)
        x = Dropout(dropout, name=block_name+'/dropout')(x)
        x = BatchNormalization(axis=3, name=block_name+'/batch_norm')(x)
        x = Activation(activation)(x)
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same', name=block_name+'/pool')(x)

    block_name = 'block_2'
    with K.name_scope(block_name) as scope:
        x = bl_bottleneck(x, filters=64, kernel_size=(3, 3), is_first_layer_of_first_block=True, block_name=block_name+'/bottleneck_1', dropout=0.0)
        x = bl_bottleneck(x, filters=64, block_name=block_name+'/bottleneck_2', dropout=0.0)

    block_name = 'block_3'
    with K.name_scope(block_name) as scope:
        x = bl_bottleneck(x, filters=128, kernel_size=(3, 3), strides=(2,2), block_name=block_name+'/bottleneck_1', dropout=0.25)
        x = bl_bottleneck(x, filters=128, block_name=block_name+'/bottleneck_2', dropout=0.25)
    source_layers.append(x)
    
    block_name = 'block_4'
    with K.name_scope(block_name) as scope:
        x = bl_bottleneck(x, filters=256, strides=(2,2), kernel_size=(3, 5), block_name=block_name+'/bottleneck_1', dropout=0.5)
        x = bl_bottleneck(x, filters=256, block_name=block_name+'/bottleneck_2', kernel_size=(3, 5), dropout=0.5)
    source_layers.append(x)
    
    block_name = 'block_5'
    with K.name_scope(block_name) as scope:
        x = bl_bottleneck(x, filters=256, strides=(2, 2), kernel_size=(3, 5), block_name=block_name+'/bottleneck_1', dropout=0.5)
        x = bl_bottleneck(x, filters=256, block_name=block_name+'/bottleneck_2', kernel_size=(3, 5), dropout=0.5)
    source_layers.append(x)

    block_name = 'block_6'
    with K.name_scope(block_name) as scope:
        x = bl_bottleneck(x, filters=512, strides=(2, 2), kernel_size=(3, 5), block_name=block_name+'/bottleneck_1', dropout=0.5)
        x = bl_bottleneck(x, filters=512, block_name=block_name+'/bottleneck_2', kernel_size=(3, 5), dropout=0.5)
    source_layers.append(x)

    block_name = 'block_7'
    with K.name_scope(block_name) as scope:
        x = bl_bottleneck(x, filters=512, strides=(2, 2), kernel_size=(3, 5), block_name=block_name+'/bottleneck_1', dropout=0.5)
        x = bl_bottleneck(x, filters=512, kernel_size=(3, 5), block_name=block_name+'/bottleneck_2', dropout=0.5)
    source_layers.append(x)

    block_name = 'block_8'
    with K.name_scope(block_name) as scope:
        x = bl_bottleneck(x, filters=1024, strides=(2, 2), kernel_size=(3, 5), block_name=block_name+'/bottleneck_1', dropout=0.5)
        x = bl_bottleneck(x, filters=1024, kernel_size=(3, 5), block_name=block_name+'/bottleneck_2', dropout=0.5)
    source_layers.append(x)

    block_name = 'block_9'
    with K.name_scope(block_name) as scope:
        x = bl_bottleneck(x, filters=1024, strides=(2, 2), kernel_size=(3, 5), block_name=block_name+'/bottleneck_1', dropout=0.5)
        x = bl_bottleneck(x, filters=1024, kernel_size=(3, 3), block_name=block_name+'/bottleneck_2', dropout=0.5)
    source_layers.append(x)
    
    return source_layers


