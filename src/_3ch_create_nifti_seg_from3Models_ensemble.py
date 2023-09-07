#Program to generate nifti output mask files on test data
from __future__ import print_function
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import nibabel as nib
from tqdm import *
import copy
from Edge_Core_Labels_WatershedAndCC import instance_seg

from _3ch_instanceCystSeg_train_unet_t001 import get_unet
#Add the .hdf5 names
modelname1 = '' # 'instanceCystSeg_modelWeights_3ch_t001'
modelname2 = ''
modelname3 = ''


#--------------------------
# END USER INPUT
#--------------------------

K.set_image_data_format('channels_first')

img_rows = 512
img_cols = 512

smooth = 1.

input_folder='FOLDER'
output_folder = 'FOLDER'
image_folder = ''
seg_folder = ''
segout_folder = 'FOLDER'      		# specify prediction output folder

oriprefix = '' 						# MR indetifier + extension
kidneyprefix = '' 					# Kidney segmentation indetifier + extension
segprefix = '_' + modelname1 + ''	# add extension
strremove = -len(kidneyprefix)
Scan = 512
count = 0

try:
	subdir, dirs, files = os.walk(input_folder).next()
	files = [k for k in files if kidneyprefix in k]
except:
	subdir, dirs, files = os.walk(input_folder).__next__()
	files = [k for k in files if kidneyprefix in k]

counter = 0

model1 = get_unet()
model1.load_weights(modelname1+'.hdf5') 
model2 = get_unet()
model2.load_weights(modelname2+'.hdf5') 
model3 = get_unet()
model3.load_weights(modelname3+'.hdf5') 
				
for filename in tqdm(files[:]):  
	count+=1
	if kidneyprefix in filename:     
		if modelname1 not in filename:
			if 'Seg' not in filename: 
				imgloaded = nib.load(input_folder+'/'+image_folder+'/'+filename[:strremove]+oriprefix)
				data = imgloaded.get_data()
				print(filename)
				print(type(data))
				data=np.asarray(data).astype(np.float32)
				data = data/np.percentile(data,99) * 255
				data[data>255] = 255

				segimg = nib.load(input_folder+'/'+image_folder+'/'+filename[:strremove]+kidneyprefix)
				segdata = segimg.get_data()
				# binarize kidney labels
				segdata[segdata>2]=0
				segdata[segdata>1]=1
				segdata = np.asarray(segdata).astype(np.float32)

				# Z-interpolation
				data_r = np.rot90(data, axes=(0,2))
				segdata_r = np.rot90(segdata, axes=(0,2))
				Scn = segdata.shape[0]
				factor = 3
				#now loop through slices			
				dataslc = np.zeros(shape=[segdata.shape[2]*factor,Scn])
				img_stack_r = np.zeros(shape=[segdata.shape[2]*factor,Scn,Scn])
				seg_stack_r = np.zeros(shape=[segdata.shape[2]*factor,Scn,Scn])

				for io in range(0,np.shape(segdata_r)[2]):
					dataslc = cv2.resize(data_r[:,:,io], (Scn, segdata.shape[2]*factor), interpolation=cv2.INTER_CUBIC)
					img_stack_r[:,:,io] = dataslc
					dataslc = cv2.resize(segdata_r[:,:,io], (Scn, segdata.shape[2]*factor), interpolation=cv2.INTER_NEAREST)
					seg_stack_r[:,:,io] = dataslc

				img_stack = np.rot90(img_stack_r, 3, axes=(0,2))
				print(np.shape(img_stack))
				seg_stack = np.rot90(seg_stack_r, 3, axes=(0,2))

				# upsample image stack to 512x512
				img_stack_512 = np.zeros(shape=[np.shape(img_stack)[2],Scan,Scan],dtype='float32') 
				for io in range(0,np.shape(img_stack)[2]):
					img_stack_512[io,:,:] = cv2.resize(img_stack[:,:,io], (Scan,Scan), interpolation=cv2.INTER_CUBIC)
				print(np.shape(img_stack_512))


				# x,y - interpolation

				dataslice = np.zeros(shape=[np.shape(img_stack)[2],4,Scan,Scan],dtype='float32') #np.zeros(shape=[np.shape(data)[2],2,Scan,Scan],dtype='float32')

				for io in range(0,np.shape(img_stack)[2]):
					if io==0:
						dataslice[io,0,:,:] = np.zeros(shape=[Scan,Scan],dtype='float32')
					else:
						dataslice[io,0,:,:] = cv2.resize(img_stack[:,:,io-1], (Scan,Scan), interpolation=cv2.INTER_CUBIC)
					dataslice[io,1,:,:] = cv2.resize(img_stack[:,:,io], (Scan,Scan), interpolation=cv2.INTER_CUBIC)
          
					if io==np.shape(seg_stack)[2]-1:
						dataslice[io,2,:,:] = np.zeros(shape=[Scan,Scan],dtype='float32')
					else:
						dataslice[io,2,:,:] = cv2.resize(img_stack[:,:,io+1], (Scan,Scan), interpolation=cv2.INTER_CUBIC)
         
					dataslice[io,3,:,:] = cv2.resize(seg_stack[:,:,io], (Scan,Scan), interpolation=cv2.INTER_NEAREST)

				img = dataslice

				print(np.shape(img))

				img = img.astype('float32')
				mean = np.mean(img[:,1,:,:])  # mean for data centering
				std = np.std(img[:,1,:,:])  # std for data normalization
				img[:,0:3,:,:] -= mean
				img[:,0:3,:,:] /= std

				print('-'*30)
				print('Predicting masks on test data...')
				print(filename)
				print('-'*30)
				print(np.min(img),np.max(img))
				print(np.mean(img),np.std(img))

				y_proba1 = model1.predict(img, batch_size=1)
				print(np.shape(y_proba1))
				print(np.sum(y_proba1[:,0,:,:]))
				print(np.sum(y_proba1[:,1,:,:]))
				print(np.sum(y_proba1[:,2,:,:]))
				imgs_mask_test1 = y_proba1.argmax(axis=1)
				edge1 = copy.deepcopy(imgs_mask_test1)
				edge1[edge1<2] = 0
				core1 = copy.deepcopy(imgs_mask_test1)
				core1[core1>1] = 0

				y_proba2 = model2.predict(img, batch_size=1)
				print(np.shape(y_proba2))
				print(np.sum(y_proba2[:,0,:,:]))
				print(np.sum(y_proba2[:,1,:,:]))
				print(np.sum(y_proba2[:,2,:,:]))
				imgs_mask_test2 = y_proba2.argmax(axis=1)
				edge2 = copy.deepcopy(imgs_mask_test2)
				edge2[edge2<2] = 0
				core2 = copy.deepcopy(imgs_mask_test2)
				core2[core2>1] = 0

				y_proba3 = model3.predict(img, batch_size=1)
				print(np.shape(y_proba3))
				print(np.sum(y_proba3[:,0,:,:]))
				print(np.sum(y_proba3[:,1,:,:]))
				print(np.sum(y_proba3[:,2,:,:]))
				imgs_mask_test3 = y_proba3.argmax(axis=1)
				edge3 = copy.deepcopy(imgs_mask_test3)
				edge3[edge3<2] = 0
				core3 = copy.deepcopy(imgs_mask_test3)
				core3[core3>1] = 0

				

				imgs_mask_test_edge = edge1 + edge2 + edge3
				print('here')
				print(np.min(imgs_mask_test_edge),np.max(imgs_mask_test_edge))
				imgs_mask_test_edge[imgs_mask_test_edge<4] = 0
				imgs_mask_test_edge[imgs_mask_test_edge>0] = 2

				imgs_mask_test_core = core1 + core2 + core3
				print('here')
				print(np.min(imgs_mask_test_core),np.max(imgs_mask_test_core))
				imgs_mask_test_core[imgs_mask_test_core<2] = 0
				imgs_mask_test_core[imgs_mask_test_core>0] = 1


				imgs_mask_test = imgs_mask_test_edge + imgs_mask_test_core
				imgs_mask_test[imgs_mask_test>2] = 2
				print(np.sum(imgs_mask_test==1))
				print(np.sum(imgs_mask_test==2))

				print(np.min(imgs_mask_test),np.max(imgs_mask_test))

				print('data shape')
				print(np.shape(data))
				
				print(np.shape(imgs_mask_test))

				affine = imgloaded.get_affine()
				info = copy.deepcopy(affine)
				print(info)
				factor1 = Scan/data.shape[1]

				info[1,2] = info[1,2]/factor


				img_EC = np.zeros(shape = [data.shape[0],data.shape[1],segdata.shape[2]*factor])
				for io in range(0,np.shape(img_stack)[2]):
					img_EC[:,:,io] = cv2.resize(imgs_mask_test[io,:,:], (data.shape[1],data.shape[0]), interpolation=cv2.INTER_NEAREST)

				img_EC = img_EC.astype(np.uint8)
				# Print upsampled edge-core map
				nifti = nib.Nifti1Image(img_EC, info)
				nifti.to_filename(input_folder+'/'+image_folder+'/'+filename[:strremove]+segprefix)

				info[0,0] = info[0,0]/factor1
				info[2,1] = info[2,1]/factor1

				imgs_mask_test_labels = instance_seg(img_stack_512, imgs_mask_test, info)

				img_out1 = np.zeros(shape = [data.shape[0],data.shape[1],segdata.shape[2]*factor])
				for io in range(0,np.shape(img_stack)[2]):
					img_out1[:,:,io] = cv2.resize(imgs_mask_test_labels[io,:,:], (np.shape(data)[1],np.shape(data)[0]), interpolation=cv2.INTER_NEAREST)


				segdataout_r = np.rot90(img_out1, axes=(0,2))
				img_out = np.zeros(shape=[segdata.shape[2],data.shape[1],data.shape[0]])
				dataslc = np.zeros(shape=[segdata.shape[2],data.shape[1]])
				for io in range(0,np.shape(segdataout_r)[2]):
					img_out[:,:,io] = cv2.resize(segdataout_r[:,:,io], (data.shape[1], segdata.shape[2]), interpolation=cv2.INTER_NEAREST)
				img_out = np.rot90(img_out, 3, axes=(0,2))

				print('here')
				print(np.min(img_out),np.max(img_out))
				print(np.shape(img_out))
				print(np.min(img_out),np.max(img_out))
				img_out = img_out.astype(np.int32)
				affine = imgloaded.get_affine()
				nifti = nib.Nifti1Image(img_out, affine)
				nifti.to_filename(input_folder+'/'+image_folder+'/'+filename[:strremove]+'_CystInstSeg.nii.gz')
					
