from __future__ import print_function
import os
import numpy as np
import cv2

#Edit data_path and the paths to numpy files
data_path = ''
train_numpy_folder = ''
val_numpy_folder = ''

image_rows = 512
image_cols = 512


def load_train_data():
    train_data_path = os.path.join(data_path, train_numpy_folder)
    images = sorted(os.listdir(train_data_path))
    total = len(images) // 2

    imgs = np.ndarray((total, 4, image_rows, image_cols), dtype=np.float32)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.float32)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.npy'
        img = np.load(os.path.join(train_data_path, image_name))

        img_mask = np.load(os.path.join(train_data_path, image_mask_name))

        imgs[i,0,:,:] = img[:,:,0]
        imgs[i,1,:,:] = img[:,:,1]
        imgs[i,2,:,:] = img[:,:,2]
        imgs[i,3,:,:] = img[:,:,3]

        imgs_mask[i,0,:,:] = img_mask


        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    return imgs, imgs_mask

def load_val_data():
    train_data_path = os.path.join(data_path, val_numpy_folder)
    images = sorted(os.listdir(train_data_path))
    total = len(images) // 2

    imgs = np.ndarray((total, 4, image_rows, image_cols), dtype=np.float32)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.float32)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.npy'
        img = np.load(os.path.join(train_data_path, image_name))

        img_mask = np.load(os.path.join(train_data_path, image_mask_name))

        imgs[i,0,:,:] = img[:,:,0]
        imgs[i,1,:,:] = img[:,:,1]
        imgs[i,2,:,:] = img[:,:,2]
        imgs[i,3,:,:] = img[:,:,3]

        imgs_mask[i,0,:,:] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    return imgs, imgs_mask

if __name__ == '__main__':
    create_train_data()
    
