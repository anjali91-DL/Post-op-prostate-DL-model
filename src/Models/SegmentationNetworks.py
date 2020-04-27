"""
#####################################################################################
Code written : 02/26/2020
Owner : Anjali Balagopal, Graduate Student, UT Southwestern medical Center
#####################################################################################
"""

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, concatenate
from tensorflow.keras.layers import Conv3D, UpSampling3D, MaxPooling3D, Dropout, BatchNormalization, Activation, Dropout, Lambda, add, Dense
from tensorflow.keras.layers import GlobalAveragePooling3D, Reshape, Dense, multiply
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import backend as K
from .DropBlock import DropBlock3D
from .groupnorm import GroupNormalization

def squeeze_excite_block(input, ratio=8):
    init = input
    channel_axis = -1  # Since we are using Tensorflow
    filters = init.shape[channel_axis]
    shape = (1, 1, 1, filters)

    se = GlobalAveragePooling3D()(init)
    se = Reshape(shape)(se)
    se = Dense(filters // ratio, activation="relu", kernel_initializer='glorot_uniform', use_bias=False)(se)
    se = Dense(filters, activation="sigmoid", kernel_initializer='glorot_uniform', use_bias=False)(se)

    output = multiply([init, se])

    return output

def residual_block3D_DB(y, nb_channels, prob, kernel = (3,3,3), _strides=(1, 1, 1), project_res=False):
    id_map = y
    residual_shape = K.int_shape(id_map)
    # down-sampling is performed with a stride of 2
    y = Conv3D(nb_channels, kernel_size=kernel, strides=(1, 1, 1), padding='same', activation='relu')(y)
    y = DropBlock3D(keep_prob= prob, block_size=3)(BatchNormalization()(y))
    y = Conv3D(nb_channels, kernel_size=kernel, strides=(1, 1, 1), padding='same')(y)
    y = DropBlock3D(keep_prob= prob, block_size=3)(BatchNormalization()(y))

    # identity shortcuts used directly when the input and output are of the same dimensions
    if project_res or _strides != (1, 1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        id_map = Conv3D(nb_channels, kernel_size=(1, 1, 1), strides=_strides, padding='same')(id_map)
        id_map = BatchNormalization()(id_map)
    print(id_map.shape, y.shape)
    y = add([id_map, y])
    y = Activation('relu')(y)

    return y

def grouped_convolution3D(y, nb_channels, cardinality, _strides):
    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        return Conv3D(nb_channels, kernel_size=(3, 3, 3), strides=_strides, padding='same')(y)
    assert not nb_channels % cardinality
    _d = nb_channels // cardinality

    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
    # and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        #print(_d, j, j * _d, j * _d + _d, y.shape[3])
        group = Lambda(lambda z: z[:, :, :, :, j * _d:j * _d + _d])(y)
        groups.append(Conv3D(_d, kernel_size=(3, 3, 3), strides=_strides, padding='same')(group))

    # the grouped convolutional layer concatenates them as the outputs of the layer
    y = concatenate(groups)

    return y

def unet_3D_SE_ResNet_DB(img_rows=None, img_cols=None, img_slcs=None, channels_in=1, channels_out=1,
                    starting_filter_number=32,
                    kernelsize=(3, 3, 3), number_of_pool=3, poolsize=(2, 2, 2), filter_rate=2,
                    final_activation='sigmoid'):
    initializer = glorot_uniform
    layer_conv = {}
    layer_others = {}

    tot_num_filters = channels_in

    number_of_layers_half = number_of_pool + 1

    number_of_filters_max = np.round((filter_rate ** (number_of_layers_half - 1)) * starting_filter_number)
    # print('max number of filters in U ' + str(number_of_filters_max))
    # print('Dropout Rate:')
    start_layer = Input((img_rows, img_cols, img_slcs, channels_in))
    layer_others[0] = Conv3D(starting_filter_number, kernel_size=(1, 1, 1), strides=(1, 1, 1))(start_layer)
    prob = 0.85
    for layer_number in range(1, number_of_layers_half):
        number_of_filters_current = np.round((filter_rate ** (layer_number - 1)) * starting_filter_number)
        prob = prob - 0.05

        init =  DropBlock3D(keep_prob=prob, block_size=3)(BatchNormalization()(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(
                layer_others[layer_number - 1])))
        if layer_number == 1:
            tot_num_filters += 2 * number_of_filters_current
        y = residual_block3D_DB(init, number_of_filters_current,prob, _strides=(1, 1, 1), project_res=False)
        # x = BatchNormalization()(Conv3D(number_of_filters_current, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='glorot_uniform')(y))

        # squeeze and excite block
        x = squeeze_excite_block(y)
        x = DropBlock3D(keep_prob=prob, block_size=3)(add([init, x]))
        layer_conv[layer_number] = Activation('relu')(x)
        layer_others[layer_number] = MaxPooling3D(pool_size=poolsize)(layer_conv[layer_number])

    # print(dropout_rate)
    layer_conv[number_of_layers_half] =  DropBlock3D(keep_prob=prob, block_size=3)(BatchNormalization()(
        Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same',
               kernel_initializer=initializer(), activation='relu')(layer_others[number_of_layers_half - 1])))

    for layer_number in range(number_of_layers_half + 1, 2 * number_of_layers_half):
        number_of_filters_current = np.round(
            (filter_rate ** (2 * number_of_layers_half - layer_number - 1)) * starting_filter_number)
        prob = prob + 0.05
        # print(drop_rate_layer)
        layer_others[layer_number] = concatenate([ DropBlock3D(keep_prob=prob, block_size=3)(BatchNormalization()(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(
                UpSampling3D(size=poolsize)(layer_conv[layer_number - 1])))),
                                                  layer_conv[2 * number_of_layers_half - layer_number]], axis=-1)
        layer_conv[layer_number] =  DropBlock3D(keep_prob=prob, block_size=3)(BatchNormalization()(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, paddunet_3D_SE_ResNet_DBing='same', activation='relu')(
                layer_others[layer_number])))
        layer_conv[layer_number] =  DropBlock3D(keep_prob=prob, block_size=3)(BatchNormalization()(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(
                layer_conv[layer_number])))

    layer_conv[2 * number_of_layers_half] = Conv3D(channels_out, kernel_size=kernelsize, padding='same',
                                                   activation=final_activation)(
        layer_conv[2 * number_of_layers_half - 1])

    model = Model(inputs=start_layer, outputs=[layer_conv[2 * number_of_layers_half]])
    return model

def unet_3D_ResNeXt_DB(img_rows=None, img_cols=None, img_slcs=None, channels_in=1, channels_out=1,
                    starting_filter_number=32, cardinality=4,
                    kernelsize=(3, 3, 3), number_of_pool=3, poolsize=(2, 2, 2), filter_rate=2,
                    final_activation='sigmoid'):
    initializer = glorot_uniform
    layer_conv = {}
    layer_others = {}

    tot_num_filters = channels_in

    number_of_layers_half = number_of_pool + 1

    number_of_filters_max = np.round((filter_rate ** (number_of_layers_half - 1)) * starting_filter_number)
    # print('max number of filters in U ' + str(number_of_filters_max))
    # print('Dropout Rate:')
    start_layer = Input((img_rows, img_cols, img_slcs, channels_in))
    layer_others[0] = Conv3D(starting_filter_number, kernel_size=(1, 1, 1), strides=(1, 1, 1))(start_layer)
    prob = 0.95
    for layer_number in range(1, number_of_layers_half):
        number_of_filters_current = np.round((filter_rate ** (layer_number - 1)) * starting_filter_number)
        prob = prob - 0.05
        # print(drop_rate_layer)

        init = DropBlock3D(keep_prob= prob, block_size=3)(BatchNormalization()(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(
                layer_others[layer_number - 1])))
        if layer_number == 1:
            tot_num_filters += 2 * number_of_filters_current
        y = grouped_convolution3D(init, number_of_filters_current, cardinality,(1,1,1))
        x = DropBlock3D(keep_prob= prob, block_size=3)(BatchNormalization()(Conv3D(number_of_filters_current, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='glorot_uniform')(y)))
        print(init.shape, x.shape)
        x = add([init, x])
        layer_conv[layer_number] = Activation('relu')(x)
        layer_others[layer_number] = MaxPooling3D(pool_size=poolsize)(layer_conv[layer_number])

    # print(dropout_rate)
    layer_conv[number_of_layers_half] = DropBlock3D(keep_prob= prob, block_size=3)(BatchNormalization()(
        Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same',
               kernel_initializer=initializer(), activation='relu')(layer_others[number_of_layers_half - 1])))

    for layer_number in range(number_of_layers_half + 1, 2 * number_of_layers_half):
        number_of_filters_current = np.round(
            (filter_rate ** (2 * number_of_layers_half - layer_number - 1)) * starting_filter_number)

        prob = prob + 0.05
        # print(drop_rate_layer)
        layer_others[layer_number] = concatenate([DropBlock3D(keep_prob= prob, block_size=3)(BatchNormalization()(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(
                UpSampling3D(size=poolsize)(layer_conv[layer_number - 1])))),
                                                  layer_conv[2 * number_of_layers_half - layer_number]], axis=-1)
        layer_conv[layer_number] = DropBlock3D(keep_prob= prob, block_size=3)(BatchNormalization()(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(
                layer_others[layer_number])))
        layer_conv[layer_number] = DropBlock3D(keep_prob= prob, block_size=3)(BatchNormalization()(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(
                layer_conv[layer_number])))

    layer_conv[2 * number_of_layers_half] = Conv3D(channels_out, kernel_size=kernelsize, padding='same',
                                                   activation=final_activation)(
        layer_conv[2 * number_of_layers_half - 1])

    model = Model(inputs=start_layer, outputs=[layer_conv[2 * number_of_layers_half]])
    return model

def unet_3D_ResNet_DB(img_rows=None, img_cols=None, img_slcs=None, channels_in=1, channels_out=1, starting_filter_number=8,
            kernelsize=(3, 3, 3), number_of_pool=5, poolsize=(2, 2,2), filter_rate=2,
            final_activation='sigmoid'):
    layer_conv = {}
    # initialize a dictionary of all other layers that are not convolution layers (e.g. input, pooling, deconv).
    layer_others = {}
    number_of_filters_current = starting_filter_number
    number_of_layers_half = number_of_pool + 1
    prob = 0.75
    number_of_filters_max = np.round((filter_rate ** (number_of_layers_half - 1)) * starting_filter_number)
    # print('max number of filters in U ' + str(number_of_filters_max))
    # first half of U
    start_layer = Input((img_rows, img_cols, img_slcs, channels_in))
    layer_others[0] = Conv3D(number_of_filters_current, kernel_size=(1, 1, 1), strides=(1, 1, 1))(start_layer)
    #pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

    for layer_number in range(1, number_of_layers_half):
        number_of_filters_current = np.round((filter_rate ** (layer_number - 1)) * starting_filter_number)

        prob = prob - 0.05
        layer_conv[layer_number] = DropBlock3D(keep_prob= prob, block_size=3)(BatchNormalization()(Conv3D(number_of_filters_current, kernel_size=(1, 1, 1), strides=(1, 1, 1))(layer_others[layer_number - 1])))
        RN1 = residual_block3D_DB(layer_conv[layer_number], number_of_filters_current,prob, kernel=(3,3,3), _strides=(1, 1, 1), project_res=False)
        RN2 = residual_block3D_DB(layer_conv[layer_number], number_of_filters_current,prob, kernel=(5,5,5), _strides=(1, 1, 1), project_res=False)


        layer_conv[layer_number] =  concatenate([RN1,RN2], axis=-1)
        # layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(residual_block3D(layer_conv[layer_number], number_of_filters_current, _strides=(1, 1, 1), project_res=False))
        # a = bottleneck_block(layer_conv[layer_number], number_of_filters_current, drop_rate_layer, _strides=(1, 1), project_res=False)
        # a = bottleneck_block(a, number_of_filters_current,drop_rate_layer, _strides=(1, 1), project_res=False)
        layer_others[layer_number] = MaxPooling3D(pool_size=poolsize)(layer_conv[layer_number])

    # center of U
    # print(dropout_rate)
    layer_conv[number_of_layers_half] = DropBlock3D(keep_prob= prob, block_size=3)(BatchNormalization()(
        Conv3D(filters=np.round((filter_rate ** (number_of_layers_half - 1)) * starting_filter_number),
               kernel_size=kernelsize, padding='same', activation='relu')(layer_others[number_of_layers_half - 1])))
    layer_conv[number_of_layers_half] = DropBlock3D(keep_prob= prob, block_size=3)(BatchNormalization()(
        Conv3D(filters=np.round((filter_rate ** (number_of_layers_half - 1)) * starting_filter_number),
               kernel_size=kernelsize, padding='same', activation='relu')(layer_conv[number_of_layers_half])))

    # second half of U
    for layer_number in range(number_of_layers_half + 1, 2 * number_of_layers_half ):
        number_of_filters_current = np.round(
            (filter_rate ** (2 * number_of_layers_half - layer_number - 1)) * starting_filter_number)
        prob = prob + 0.05
        # print(drop_rate_layer)
        layer_others[layer_number] = concatenate([(BatchNormalization()(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(
                UpSampling3D(size=poolsize)(layer_conv[layer_number - 1])))),
                                                  layer_conv[2 * number_of_layers_half - layer_number]], axis=-1)

        layer_conv[layer_number] = DropBlock3D(keep_prob= prob, block_size=3)(BatchNormalization()(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(
                layer_others[layer_number])))
        layer_conv[layer_number] = DropBlock3D(keep_prob= prob, block_size=3)(BatchNormalization()(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(
                layer_conv[layer_number])))
        print(layer_number)

    layer_conv[2 * number_of_layers_half] = Conv3D(channels_out, kernel_size=kernelsize, padding='same', activation=final_activation)(layer_conv[2 * number_of_layers_half - 1])

    # build and compile U
    model = Model(inputs= start_layer, outputs=[layer_conv[2 * number_of_layers_half]])
    return model

def AGMTN(img_rows=None, img_cols=None, img_slcs=None, channels_in=1, channels_out=1,
                    starting_filter=32, gn_param=32, number_of_pool=3, poolsize=(2, 2, 2)):
    initializer = glorot_uniform
    layer_conv = {}
    layer_upconv_decodermask = {}
    layer_upconv_decoderdistance = {}
    layer_others = {}
    layer_others_decodermask = {}
    layer_others_decoderdistance = {}
    kernelsize1 = (5, 5, 5)
    kernelsize2 = (3, 3, 3)
    n = 0
    number_of_layers_half = number_of_pool + 1
    prob = 1.0
    number_of_filters_max = np.round((2 ** (number_of_layers_half - 1)) * starting_filter)
    layer_others[0] = Input((img_rows, img_cols, img_slcs, channels_in))
    for layer_number in range(1, number_of_layers_half):
        number_of_filters_current = np.round((2 ** (layer_number - 1)) * starting_filter)
        prob = prob-.05
        groups = int(
            np.clip(np.round((2 ** (number_of_layers_half - 1)) * starting_filter) / gn_param, 1,
                    np.round((2 ** (number_of_layers_half - 1)) * starting_filter)))
        Incept1 = DropBlock3D(keep_prob=prob, block_size=3)(GroupNormalization(groups=groups)(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize1, padding='same',
                   kernel_initializer=initializer(), activation='relu')(layer_others[layer_number - 1])))
        Incept2 = DropBlock3D(keep_prob=prob, block_size=3)(GroupNormalization(groups=groups)(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize2, padding='same',
                   kernel_initializer=initializer(), activation='relu')(layer_others[layer_number - 1])))
        layer_conv[layer_number] = DropBlock3D(keep_prob=prob, block_size=3)(concatenate([Incept1, Incept2], axis=-1))
        layer_others[layer_number] = DropBlock3D(keep_prob=prob, block_size=3)(
            MaxPooling3D(pool_size=poolsize)(layer_conv[layer_number]))
    # print(dropout_rate)
    number_of_filters_current = np.round(
        (2 ** (2 * number_of_layers_half - number_of_layers_half - 1)) * starting_filter)

    Incept1 = DropBlock3D(keep_prob=prob, block_size=3)(GroupNormalization(groups=groups)(
        Conv3D(filters=number_of_filters_current, kernel_size=kernelsize1, padding='same',
               kernel_initializer=initializer(), activation='relu')(layer_others[number_of_layers_half - 1])))
    Incept2 = DropBlock3D(keep_prob=prob, block_size=3)(GroupNormalization(groups=groups)(
        Conv3D(filters=number_of_filters_current, kernel_size=kernelsize2, padding='same',
               kernel_initializer=initializer(), activation='relu')(layer_others[number_of_layers_half - 1])))

    layer_conv[number_of_layers_half] = DropBlock3D(keep_prob=prob, block_size=3)(concatenate([Incept1, Incept2], axis=-1))
    layer_upconv_decodermask[number_of_layers_half] = layer_conv[number_of_layers_half]

    layer_upconv_decoderdistance[number_of_layers_half] = layer_conv[number_of_layers_half]
    for layer_number in range(number_of_layers_half + 1, 2 * number_of_layers_half):
        n += 1
        number_of_filters_current = np.round(
            (2 ** (2 * number_of_layers_half - layer_number - 1)) * starting_filter)
        groups = int(
            np.clip(np.round((2 ** (number_of_layers_half - 1)) * starting_filter) / gn_param, 1,
                    np.round((2 ** (number_of_layers_half - 1)) * starting_filter)))
        prob = prob +.05
        #####DECODER DISTANCE
        layer_others_decoderdistance[layer_number] = UpSampling3D(size=poolsize)(
            layer_upconv_decoderdistance[layer_number - 1])
        layer_upconv_decoderdistance[layer_number] = DropBlock3D(keep_prob=prob, block_size=3)(GroupNormalization(groups=groups)(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize2, padding='same', activation='relu')(
                layer_others_decoderdistance[layer_number])))
        layer_upconv_decoderdistance[layer_number] = DropBlock3D(keep_prob=prob, block_size=3)(GroupNormalization(groups=groups)(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize2, padding='same', activation='relu')(
                layer_upconv_decoderdistance[layer_number])))

        AG = layer_conv[2 * number_of_layers_half - layer_number]
        layer_others_decodermask[layer_number] = DropBlock3D(keep_prob=prob, block_size=3)(concatenate([(GroupNormalization(groups=groups)(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize2, padding='same', activation='relu')(
                UpSampling3D(size=poolsize)(layer_upconv_decodermask[layer_number - 1])))), AG], axis=-1))
        layer_upconv_decodermask[layer_number] = DropBlock3D(keep_prob=prob, block_size=3)(GroupNormalization(groups=groups)(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize2, padding='same', activation='relu')(
                layer_others_decodermask[layer_number])))
        layer_upconv_decodermask[layer_number] = DropBlock3D(keep_prob=prob, block_size=3)(GroupNormalization(groups=groups)(
            Conv3D(filters=number_of_filters_current, kernel_size=kernelsize2, padding='same', activation='relu')(
                layer_upconv_decodermask[layer_number])))

    out3 = Conv3D(channels_out, kernel_size=kernelsize2, padding='same', activation='sigmoid')(
       UpSampling3D(size=(4, 4, 4))(layer_upconv_decodermask[2 * number_of_layers_half - 3]))
    out4 = Conv3D(channels_out, kernel_size=kernelsize2, padding='same', activation='sigmoid')(
        UpSampling3D(size=(2, 2, 2))(layer_upconv_decodermask[2 * number_of_layers_half -2]))
    layer_upconv_decodermask[2 * number_of_layers_half] = Conv3D(channels_out, kernel_size=kernelsize2,
                                                                 padding='same', name='dmbeforesigmoid')(
        layer_upconv_decodermask[2 * number_of_layers_half - 1])
    decode_mask = Activation('sigmoid', name='dm')(layer_upconv_decodermask[2 * number_of_layers_half])
    layer_upconv_decoderdistance[2 * number_of_layers_half] = Conv3D(channels_out, kernel_size=kernelsize2,
                                                                     padding='same'
                                                                     , name='dd')(
        layer_upconv_decoderdistance[2 * number_of_layers_half - 1])
    model = Model(inputs=[layer_others[0]],
                  outputs=[decode_mask, layer_upconv_decoderdistance[2 * number_of_layers_half], out3, out4])

    return model





