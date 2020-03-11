#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script calculates an error between two segmentations for 13284 sample 
@author: Daniil Kazantsev
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

path_folder = "/dls/i23/data/2019/nr23571-5/processing/tomography/recon/13284/PAPER_PROC/avizo/tiffs_xz/"
imagename = "13284_segment_xz_"

tot_num_im = 1000
start_index = '0000'
image = Image.open(path_folder + imagename + start_index + ".tif")
imarray = np.array(image)
[dimX,dimY] = np.shape(imarray)

volume_manual = np.zeros((tot_num_im, dimX, dimY)).astype(np.uint8)

for i in range(0,tot_num_im):
    if (i < 10):
        index_str = '000' + str(i)
    elif (10 <= i < 100):
        index_str = '00' + str(i)
    elif (100 <= i < 1000):
        index_str = '0' + str(i)
    else:
        index_str = str(i)
    volume_manual[i,:,:] = np.array(Image.open(path_folder + imagename + index_str + ".tif"))


fig = plt.figure(1)
fig.suptitle('Manual 4 phases segmentation in Avizo')
plt.subplot(131)
plt.imshow(volume_manual[450,:,:],vmin=0, vmax=3)
plt.title('axial view')

plt.subplot(132)
plt.imshow(volume_manual[:,250,:],vmin=0, vmax=3)
plt.title('coronal view')

plt.subplot(133)
plt.imshow(volume_manual[:,:,200],vmin=0, vmax=3)
plt.title('sagittal view')
fig.set_size_inches((13, 8), forward=False)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()
fig.savefig('13284_manual.png', dpi=350)
#%%
# Getting automatically segmented data
#path_folder = "/dls/i23/data/2019/nr23571-5/processing/tomography/recon/13284/PAPER_PROC/MaskEvolve/Crystal/FULL_SEGMENTATION/ImageSaver-SEGMENTED_4phases/"
path_folder = "/dls/i23/data/2019/nr23571-5/processing/tomography/recon/13284/PAPER_PROC/GeoDist/FINAL_SEGM/ImageSaver-SEGMENTED_4phases/"
imagename = "SEGMENTED_4phases_13284_processed_processed_processed_processed_processed_"

volume_automatic = np.zeros((tot_num_im, dimX, dimY)).astype(np.uint8)

for i in range(0,tot_num_im):
    if (i < 10):
        index_str = '0000' + str(i)
    elif (10 <= i < 100):
        index_str = '000' + str(i)
    elif (100 <= i < 1000):
        index_str = '00' + str(i)
    else:
        index_str = str(i)
    volume_automatic[i,:,:] = np.flipud(np.array(Image.open(path_folder + imagename + index_str + ".tif")))

# change classes to fit manual segmentation:
volume_automatic[0:29,:,:] = 0
volume_automatic[volume_automatic == 5] = 0 # vacuum
volume_automatic[volume_automatic == 255] = 3 # liquor
volume_automatic[volume_automatic == 186] = 1 # loop
volume_automatic[volume_automatic == 200] = 2 # crystal

fig = plt.figure(2)
fig.suptitle('Automatic 4 phases segmentation GMM + Geodistance (crystal)')

plt.subplot(131)
plt.imshow(volume_automatic[450,:,:],vmin=0, vmax=3)
plt.title('axial view')

plt.subplot(132)
plt.imshow(volume_automatic[:,250,:],vmin=0, vmax=3)
plt.title('coronal view')

plt.subplot(133)
plt.imshow(volume_automatic[:,:,200],vmin=0, vmax=3)
plt.title('sagittal view')
fig.set_size_inches((13, 8), forward=False)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()
fig.savefig('13284geodistance.png', dpi=350)
#%%
# Compare the result of segmentation (quantitative approach for each class separetly)
class_to_calculate_error = 2 # identify the class of interest
segm_class_man = np.zeros((tot_num_im, dimX, dimY)).astype(np.int8)
segm_class_auto = np.zeros((tot_num_im, dimX, dimY)).astype(np.int8)
segm_class_man[volume_manual == class_to_calculate_error] = 1
segm_class_auto[volume_automatic == class_to_calculate_error] = 1

segm_diff = abs(segm_class_man - segm_class_auto)

fig = plt.figure(8)
fig.suptitle('Error between manual and Geodistance segmentation (crystal)')

plt.subplot(131)
plt.imshow(segm_diff[450,:,:], vmin=0, vmax=1)
plt.title('segmentation error, axial view')

plt.subplot(132)
plt.imshow(segm_diff[:,250,:], vmin=0, vmax=1)
plt.title('segmentation error, coronal view')

plt.subplot(133)
plt.imshow(segm_diff[:,:,200], vmin=0, vmax=1)
plt.title('segmentation error, sagittal view')
fig.set_size_inches((13, 8), forward=False)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()
fig.savefig('13284_geodistance_error.png', dpi=350)

# plotting the error for each slice
vector_manual = np.zeros((tot_num_im)).astype(np.int64)
vector_automatic = np.zeros((tot_num_im)).astype(np.int64)
vector_errors = np.zeros((tot_num_im)).astype(np.int64)
for i in range(0,tot_num_im):
    vector_errors[i] = np.count_nonzero(segm_diff[i,:,:])
    vector_manual[i] = np.count_nonzero(segm_class_man[i,:,:])
    vector_automatic[i] = np.count_nonzero(segm_class_auto[i,:,:])
#%%
# visualise the result    
ind = np.arange(tot_num_im) 
barWidth = 1.5
names = ['Manual','Automatic']

fig = plt.figure(10)
plt.rcParams.update({'font.size': 15})
p1 = plt.bar(ind, vector_manual,  alpha = 0.7, color='r', width=barWidth)
p2 = plt.bar(ind, vector_automatic, alpha = 0.25, color='b', width=barWidth)
plt.xlim([300,800])
#plt.yticks(fontsize=12)
plt.ylabel('The total number of pixels', fontsize=14)
plt.xlabel('2D (x-y) slices', fontsize=14)
plt.legend((p1[0], p2[0]), (names[0], names[1]), fontsize=16, ncol=2, framealpha=0, fancybox=True)
plt.title('Geodistance and manual segementation discrepancies for a crystal')
plt.show()
#fig = plt.gcf()
fig.set_size_inches((9, 11), forward=False)
fig.savefig('13284_CRYSTAL_geodistance.png', dpi=350)
#%%