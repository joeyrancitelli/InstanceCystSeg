import sys
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda, concatenate, add
from keras.layers import ELU, LeakyReLU
from metric import dice_coef, dice_coef_loss

IMG_ROWS, IMG_COLS = 512,512

def _shortcut(_input, residual):
    stride_width = _input._keras_shape[2] / residual._keras_shape[2]
    stride_height = _input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == _input._keras_shape[1]

    shortcut = _input

    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 kernel_initializer="he_normal", padding="valid")(_input)

    return add([shortcut, residual]) 

def inception_block(inputs, depth, batch_mode=0, splitted=False, activation='relu'):
    assert depth % 16 == 0
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None
    
    c1_1 = Conv2D(depth//4, (1, 1), kernel_initializer='he_normal', padding='same')(inputs)
    
    c2_1 = Conv2D(depth//8*3, (1, 1), kernel_initializer='he_normal', padding='same')(inputs)
    c2_1 = actv()(c2_1)
    if splitted:
        c2_2 = Conv2D(depth//2, (1, 3), kernel_initializer='he_normal', padding='same')(c2_1)
        c2_2 = BatchNormalization(axis=1)(c2_2)
        c2_2 = actv()(c2_2)
        c2_3 = Conv2D(depth//2, (3, 1), kernel_initializer='he_normal', padding='same')(c2_2)
    else:
        c2_3 = Conv2D(depth//2, (3, 3), kernel_initializer='he_normal', padding='same')(c2_1)
    
    c3_1 = Conv2D(depth//16, (1, 1), kernel_initializer='he_normal', padding='same')(inputs)

    c3_1 = actv()(c3_1)
    if splitted:
        c3_2 = Conv2D(depth//8, (1, 5), kernel_initializer='he_normal', padding='same')(c3_1)
        c3_2 = BatchNormalization(axis=1)(c3_2)
        c3_2 = actv()(c3_2)
        c3_3 = Conv2D(depth//8, (5, 1), kernel_initializer='he_normal', padding='same')(c3_2)
    else:
        c3_3 = Conv2D(depth//8, (5, 5), kernel_initializer='he_normal', padding='same')(c3_1)
    
    p4_1 = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(inputs)
    c4_2 = Conv2D(depth//8, (1, 1), kernel_initializer='he_normal', padding='same')(p4_1)
    
    res = concatenate([c1_1, c2_3, c3_3, c4_2], axis=1)
    res = BatchNormalization(axis=1)(res)
    res = actv()(res)
    return res
    

def rblock(inputs, num, depth, scale=0.1):    
    residual = Conv2D(depth, (num, num), padding='same')(inputs)
    residual = BatchNormalization(axis=1)(residual)
    residual = Lambda(lambda x: x*scale)(residual)
    res = _shortcut(inputs, residual)
    return ELU()(res) 
    

def NConvolution2D(nb_filter, nb_row, nb_col, padding='same', subsample=(1, 1)):
    def f(_input):
        conv = Conv2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                              padding=padding)(_input)
        norm = BatchNormalization(axis=1)(conv)
        return ELU()(norm)

    return f
    