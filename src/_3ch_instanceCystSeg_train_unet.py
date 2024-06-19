from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, add
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from unet_inception import NConvolution2D, rblock, inception_block, _shortcut
from metric import dice_coef, dice_coef_loss,jaccard_distance_loss
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from _3ch_instanceCystSeg_data_prepare_ALL_2d_n512 import load_train_data, load_val_data

K.set_image_data_format('channels_first') 

img_rows = 512
img_cols = 512

smooth = 1.
do = 0.1

def get_unet():
    splitted = False
    act = 'elu'

    inputs = Input((4, img_rows, img_cols))  

    conv1 = inception_block(inputs, 64, activation=act)
    
    pool1 = NConvolution2D(64, 3, 3, padding='same', subsample=(2,2))(conv1)
    pool1 = Dropout(do)(pool1)
    
    conv2 = inception_block(pool1, 128, activation=act)
    pool2 = NConvolution2D(128, 3, 3, padding='same', subsample=(2,2))(conv2)
    pool2 = Dropout(do)(pool2)
    
    conv3 = inception_block(pool2, 256, activation=act)
    pool3 = NConvolution2D(256, 3, 3, padding='same', subsample=(2,2))(conv3)
    pool3 = Dropout(do)(pool3)
     
    conv4 = inception_block(pool3, 512, activation=act)
    pool4 = NConvolution2D(512, 3, 3, padding='same', subsample=(2,2))(conv4)
    pool4 = Dropout(do)(pool4)
    
    conv5 = inception_block(pool4, 1024, activation=act)
    conv5 = Dropout(do)(conv5)
    
    after_conv4 = rblock(conv4, 1, 512)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), after_conv4], axis=1)
    conv6 = inception_block(up6, 512, activation=act)
    conv6 = Dropout(do)(conv6)
    
    after_conv3 = rblock(conv3, 1, 256)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), after_conv3], axis=1)
    conv7 = inception_block(up7, 256, activation=act)
    conv7 = Dropout(do)(conv7)
    
    after_conv2 = rblock(conv2, 1, 128)
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), after_conv2], axis=1)
    conv8 = inception_block(up8, 128, activation=act)
    conv8 = Dropout(do)(conv8)
    
    after_conv1 = rblock(conv1, 1, 64)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), after_conv1], axis=1)
    conv9 = inception_block(up9, 64, activation=act)
    conv9 = Dropout(do)(conv9)
    
    conv12 = Conv2D(3, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv12)

    model.compile(loss=jaccard_distance_loss,optimizer=Adam(learning_rate=1.E-3,decay=1.E-5),metrics=[jaccard_distance_loss])

    return model

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train[:,1,:,:])  # mean for data centering
    std = np.std(imgs_train[:,1,:,:])  # std for data normalization

    imgs_train[:,0:3,:,:] -= mean
    imgs_train[:,0:3,:,:] /= std

    print('mean = '+str(np.mean(imgs_train)))
    print('SD = '+str(np.std(imgs_train)))

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('uint8')

    print('mask values for train')
    print(np.min(imgs_mask_train),np.max(imgs_mask_train))

    new_train_mask = np.zeros(shape=(np.shape(imgs_mask_train)[0],3,img_cols,img_rows))

    new_train_mask[:,0:1,:,:] = imgs_mask_train==0
    new_train_mask[:,1:2,:,:] = imgs_mask_train==1
    new_train_mask[:,2:3,:,:] = imgs_mask_train==2

    imgs_mask_train = new_train_mask
    imgs_mask_train = imgs_mask_train.astype('uint8')
    
    print(np.sum(new_train_mask[:,0,:,:]),np.sum(new_train_mask[:,1,:,:]),
        np.sum(new_train_mask[:,2,:,:]))

    print('-'*30)
    print('Loading and preprocessing val data...')
    print('-'*30)
    imgs_val, imgs_mask_val = load_val_data()

    imgs_val = imgs_val.astype('float32')
    mean = np.mean(imgs_val[:,1,:,:])  # mean for data centering
    std = np.std(imgs_val[:,1,:,:])  # std for data normalization

    imgs_val[:,0:3,:,:] -= mean
    imgs_val[:,0:3,:,:] /= std

    print('mean = '+str(np.mean(imgs_val)))
    print('SD = '+str(np.std(imgs_val)))

    imgs_mask_val = imgs_mask_val.astype('float32')
    imgs_mask_val = imgs_mask_val.astype('uint8')

    print('mask values for val')
    print(np.min(imgs_mask_val),np.max(imgs_mask_val))
    new_val_mask = np.zeros(shape=(np.shape(imgs_mask_val)[0],3,img_cols,img_rows))

    new_val_mask[:,0:1,:,:] = imgs_mask_val==0
    new_val_mask[:,1:2,:,:] = imgs_mask_val==1
    new_val_mask[:,2:3,:,:] = imgs_mask_val==2

    imgs_mask_val = new_val_mask
    imgs_mask_val = imgs_mask_val.astype('uint8')

    print(np.sum(new_val_mask[:,0,:,:]),np.sum(new_val_mask[:,1,:,:]),
        np.sum(new_val_mask[:,2,:,:]))

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()

    model_checkpoint = ModelCheckpoint('instanceCystSeg_modelWeights_3ch.hdf5', monitor='val_loss',
        save_best_only=True,
        verbose = True)
    print(model.summary())
    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    print('mean train = '+str(np.mean(imgs_train)))
    print('SD train = '+str(np.std(imgs_train)))
    print('mean val = '+str(np.mean(imgs_val)))
    print('SD val = '+str(np.std(imgs_val)))

    class_weight = {0: 1.,
                1: 100.,
                2: 200.}

    model.fit(imgs_train,imgs_mask_train, validation_data = (imgs_val,imgs_mask_val), batch_size=6, 
        nb_epoch=200, verbose=1, shuffle=True,
              callbacks=[model_checkpoint])              

if __name__ == '__main__':
    train_and_predict()
