"""
#####################################################################################
Code written : 02/26/2020
Owner : Anjali Balagopal, Graduate Student, UT Southwestern medical Center
#####################################################################################
"""
import numpy as np
import tensorflow
from tensorflow.keras import backend as K

ep = 1e-7

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + ep) / (K.sum(y_true_f) + K.sum(y_pred_f) + ep)

def RS_dice_coef(y_true, y_pred):
    ep = 1e-7
    y_true_f = K.flatten(y_true)
    y_pred_f = K.sqrt(K.flatten(y_pred)+ ep)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

def weighted_dice_coef(y_true, y_pred):
    k_width = 24
    w_max = 1.
    edge = K.abs(
        K.conv3d(y_true, np.float32([[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]).reshape((3, 3, 1, 1, 1)), padding='same',
                 data_format='channels_last'))
    gk = w_max * np.ones((k_width, k_width, 1, 1, 1), dtype='float32') / 4.
    x_edge = K.clip(K.conv3d(edge, gk, padding='same', data_format='channels_last'), 0., w_max)
    w_f = K.flatten(x_edge + 1.)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(w_f * y_true_f * y_pred_f)
    return (2. * intersection + ep) / (K.sum(w_f * y_true_f) + K.sum(w_f * y_pred_f) + ep)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def channel_Dice(y_true, y_pred):
    y_true_1 = y_true[:, :, :, 0]
    y_true_2 = y_true[:, :, :, 1]
    y_true_3 = y_true[:, :, :, 2]
    y_true_4 = y_true[:, :, :, 3]
    y_true_5 = y_true[:, :, :, 4]

    y_pred_1 = y_pred[:, :, :, 0]
    y_pred_2 = y_pred[:, :, :, 1]
    y_pred_3 = y_pred[:, :, :, 2]
    y_pred_4 = y_pred[:, :, :, 3]
    y_pred_5 = y_pred[:, :, :, 4]

    Loss_1 = dice_coef(y_true_1, y_pred_1)
    Loss_2 = dice_coef(y_true_2, y_pred_2)
    Loss_3 = dice_coef(y_true_3, y_pred_3)
    Loss_4 = dice_coef(y_true_4, y_pred_4)
    Loss_5 = dice_coef(y_true_5, y_pred_5)
    return Loss_1 * 0.1 + Loss_2 * 0.3 + Loss_3 * 0.1 + Loss_4 * 0.1 + Loss_5 * 0.3

def channel_loss(y_true, y_pred):
    return -channel_Dice(y_true, y_pred)

def channel_RSDice(y_true, y_pred):
    y_true_1 = y_true[:, :, :, 0]
    y_true_2 = y_true[:, :, :, 1]
    y_true_3 = y_true[:, :, :, 2]
    y_true_4 = y_true[:, :, :, 3]
    y_true_5 = y_true[:, :, :, 4]

    y_pred_1 = y_pred[:, :, :, 0]
    y_pred_2 = y_pred[:, :, :, 1]
    y_pred_3 = y_pred[:, :, :, 2]
    y_pred_4 = y_pred[:, :, :, 3]
    y_pred_5 = y_pred[:, :, :, 4]

    Loss_1 = RS_dice_coef(y_true_1, y_pred_1)
    Loss_2 = RS_dice_coef(y_true_2, y_pred_2)
    Loss_3 = RS_dice_coef(y_true_3, y_pred_3)
    Loss_4 = RS_dice_coef(y_true_4, y_pred_4)
    Loss_5 = RS_dice_coef(y_true_5, y_pred_5)
    return Loss_1 * 0.1 + Loss_2 * 0.3 + Loss_3 * 0.1 + Loss_4 * 0.1 + Loss_5 * 0.3

def channel_RSloss(y_true, y_pred):
    return -channel_RSDice(y_true, y_pred)

def weighted_binary_crossentropy(y_true, y_pred):
    zero_weight = 0.11
    one_weight = 0.89
    # Original binary crossentropy (see losses.py):
    # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

    # Calculate the binary crossentropy
    b_ce = K.binary_crossentropy(y_true, y_pred)

    # Apply the weights
    weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
    weighted_b_ce = weight_vector * b_ce

    # Return the mean error
    return K.mean(weighted_b_ce)

def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) +
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_dice_loss2D(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
            y_true, pool_size=(9, 9), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 6. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_dice_loss(y_true, y_pred, weight)
    return loss

def weighted_bce_dice_loss3D(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool3d(
            y_true, pool_size=(20, 20, 1), strides=(1, 1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_dice_loss(y_true, y_pred, weight) + weighted_bce_loss(y_true, y_pred, weight)
    return loss

def weighted_dice_loss3D(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool3d(
            y_true, pool_size=(9, 9 , 9), strides=(1, 1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 6. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_dice_loss(y_true, y_pred, weight)
    return loss

