import os
import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Activation, Dropout

def unet_2D(img_rows=None, img_cols=None, channels_in=1, channels_out=1, starting_filter_number=8,
            kernelsize=(3, 3), number_of_pool=5, poolsize=(2, 2), filter_rate=2, dropout_rate=0.5,
            final_activation='sigmoid'):
    layer_conv = {}
    # initialize a dictionary of all other layers that are not convolution layers (e.g. input, pooling, deconv).
    layer_others = {}

    number_of_layers_half = number_of_pool + 1
    drop_rate_layer = dropout_rate
    number_of_filters_max = np.round((filter_rate ** (number_of_layers_half - 1)) * starting_filter_number)
    # print('max number of filters in U ' + str(number_of_filters_max))

    # first half of U
    layer_others[0] = Input((img_rows, img_cols, channels_in))
    for layer_number in range(1, number_of_layers_half):
        number_of_filters_current = np.round((filter_rate ** (layer_number - 1)) * starting_filter_number)
        drop_rate_layer = drop_rate_layer + 0.1
        # print(drop_rate_layer)
        layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(BatchNormalization()(
            Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(
                layer_others[layer_number - 1])))
        layer_conv[layer_number] = (BatchNormalization()(
            Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(
                layer_conv[layer_number])))
        layer_others[layer_number] = MaxPooling2D(pool_size=poolsize)(layer_conv[layer_number])

    # center of U
    # print(dropout_rate)
    layer_conv[number_of_layers_half] = Dropout(rate=dropout_rate)(BatchNormalization()(
        Conv2D(filters=np.round((filter_rate ** (number_of_layers_half - 1)) * starting_filter_number),
               kernel_size=kernelsize, padding='same', activation='relu')(layer_others[number_of_layers_half - 1])))
    layer_conv[number_of_layers_half] = Dropout(rate=dropout_rate)(BatchNormalization()(
        Conv2D(filters=np.round((filter_rate ** (number_of_layers_half - 1)) * starting_filter_number),
               kernel_size=kernelsize, padding='same', activation='relu')(layer_conv[number_of_layers_half])))

    # second half of U
    for layer_number in range(number_of_layers_half + 1, 2 * number_of_layers_half):
        number_of_filters_current = np.round(
            (filter_rate ** (2 * number_of_layers_half - layer_number - 1)) * starting_filter_number)
        drop_rate_layer = drop_rate_layer - 0.1
        # print(drop_rate_layer)
        layer_others[layer_number] = concatenate([Conv2DTranspose(number_of_filters_current, kernel_size=kernelsize,
                                                                  strides=(2, 2), kernel_initializer='glorot_uniform',
                                                                  padding='same')(layer_conv[layer_number - 1]),
                                                  layer_conv[2 * number_of_layers_half - layer_number]], axis=3)
        layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(BatchNormalization()(
            Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(
                layer_others[layer_number])))
        layer_conv[layer_number] = (BatchNormalization()(
            Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(
                layer_conv[layer_number])))

    layer_conv[2 * number_of_layers_half] = Conv2D(channels_out, kernel_size=kernelsize, padding='same',
                                                   activation=final_activation)(
        layer_conv[2 * number_of_layers_half - 1])

    # build and compile U
    model = Model(inputs=[layer_others[0]], outputs=[layer_conv[2 * number_of_layers_half]])
    return model


