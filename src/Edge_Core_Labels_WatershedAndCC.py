import numpy as np
import nibabel as nib
import skimage
import os
import copy

from skimage import measure 
from skimage import morphology
from scipy import ndimage
from collections import Counter
from skimage.morphology import watershed
from skimage.feature import peak_local_max

def instance_seg(image_data, semantic_seg, affine):

    cystsz = 20 # approx. cyst diameter in mm   
    x = int(np.round(cystsz/np.abs(affine[0,0])))
    y = int(np.round(cystsz/np.abs(affine[2,1])))
    z = int(np.round(cystsz/np.abs(affine[1,2])))

    cystdata = copy.deepcopy(semantic_seg)    
    cystdata[cystdata>1]=0

    # Watershed segmentation to separate connected cysts
    distanceI = image_data #image-based
    distanceS = ndimage.distance_transform_edt(cystdata) #shape-based
    distance = (distanceI/np.max(distanceI)) * (distanceS/np.max(distanceS)) #weighting distance with both
    coords = peak_local_max(distance, footprint=np.ones((z, x, y)), labels=cystdata) 
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels_ws = watershed(-distance, markers, mask=cystdata)

    #ConnectedComponents
    ws_copy = copy.deepcopy(labels_ws)
    ws_copy[ws_copy>0]=1
    labels_cc = cystdata - ws_copy
    labels_cc = measure.label(labels_cc,connectivity=1)
    labels_cc = np.where(labels_cc, labels_cc + lastLabel, labels_cc)
    
    labels_ws_cc = labels_ws + labels_cc
    
    # label-by label dilate 1voxel-3D (smallest to largest)
    labels2 = np.asarray(labels_ws_cc).reshape(-1)
    labels2 = labels2[labels2!=0]
    c = Counter(labels2)
    index = sorted(c.items(), key=lambda i: i[1])
    cystDilation = np.zeros(shape=(cystdata.shape[0],cystdata.shape[1],cystdata.shape[2]))
    final_labels = np.zeros(shape=(cystdata.shape[0],cystdata.shape[1],cystdata.shape[2]))
    for i in index:
        if i[1]>1:
            cystCore = copy.deepcopy(labels_ws_cc)
            cystCore[cystCore!=i[0]]=0
            cystCore[cystCore==i[0]]=1
            cystDilation = morphology.binary_dilation(cystCore)
            cystDilation = np.multiply(cystDilation, 1)
            cystDilation[cystDilation>0]=i[0]
            final_labels = np.where(cystDilation != 0, cystDilation, final_labels)
            f_labels = final_labels.astype(int)
    cystNum2 = np.unique(f_labels)
    cystNum2 = cystNum2[cystNum2!=0]
    print('Cyst count:')
    print(np.size(cystNum2))
   
    f_labels = f_labels.astype(np.int32)
    return f_labels