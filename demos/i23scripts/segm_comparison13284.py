#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script calculates an error between two segmentations for 13284 sample 
@author: Daniil Kazantsev
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

tot_num_im = 395
# read reconstructed images
path_folder = "/media/algol/F2FE9B0BFE9AC76F/__I23_data__/13284/recon/tiffs/cropped/"
imagename = "recon_"
start_index = '00000'
image = Image.open(path_folder + imagename + start_index + ".tif")
imarray = np.array(image)
[dimX,dimY] = np.shape(imarray)

volume_recon = np.zeros((tot_num_im, dimX, dimY)).astype(np.float32)

for i in range(0,tot_num_im):
    if (i < 10):
        index_str = '0000' + str(i)
    elif (10 <= i < 100):
        index_str = '000' + str(i)
    elif (100 <= i < 1000):
        index_str = '00' + str(i)
    else:
        index_str = str(i)
    volume_recon[i,:,:] = np.array(Image.open(path_folder + imagename + index_str + ".tif"))

fig = plt.figure(1)
#fig.suptitle('TomoBAR reconstructed image')
ax = fig.add_subplot(131)
plt.imshow(volume_recon[:,550,:],vmin=0, vmax=255,  cmap="gray")
ax.set_aspect(1.3)
plt.title('axial view')

ax = fig.add_subplot(132)
plt.imshow(volume_recon[160,:,:],vmin=0, vmax=255, cmap="gray")
ax.set_aspect(0.75)
plt.title('coronal view')

ax = fig.add_subplot(133)
plt.imshow(volume_recon[:,:,230],vmin=0, vmax=255,  cmap="gray")
ax.set_aspect(1.5)
plt.title('sagittal view')
fig.set_size_inches((13, 8), forward=False)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()
fig.savefig('13284_tomobar_recon.png', dpi=250)
#%%
path_folder = "/media/algol/F2FE9B0BFE9AC76F/__I23_data__/13284/avizo/converted/"
imagename = "13284_thresholding_magic_wand.labels"

tot_num_im = 395
start_index = '000'
image = Image.open(path_folder + imagename + start_index + ".tif")
imarray = np.array(image)
[dimX,dimY] = np.shape(imarray)

volume_manual = np.zeros((tot_num_im, dimX, dimY)).astype(np.uint8)

for i in range(0,tot_num_im):
    if (i < 10):
        index_str = '00' + str(i)
    elif (10 <= i < 100):
        index_str = '0' + str(i)
    else:
        index_str = str(i)
    volume_manual[i,:,:] = np.array(Image.open(path_folder + imagename + index_str + ".tif"))


fig = plt.figure(2)
#fig.suptitle('Manual 4 phases segmentation in Avizo')
ax = fig.add_subplot(131)
plt.imshow(volume_manual[:,550,:],vmin=0, vmax=3)
ax.set_aspect(1.3)
#plt.title('axial view')
plt.axis('off')

ax = fig.add_subplot(132)
plt.imshow(volume_manual[160,:,:],vmin=0, vmax=3)
ax.set_aspect(0.75)
#plt.title('coronal view')
plt.axis('off')

ax = fig.add_subplot(133)
plt.imshow(volume_manual[:,:,230],vmin=0, vmax=3)
ax.set_aspect(1.5)
#plt.title('sagittal view')
plt.axis('off')
fig.set_size_inches((13, 8), forward=False)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()
fig.savefig('13284_manual.png', dpi=250)
#%%
# Getting automatically segmented data
path_folder = "/media/algol/F2FE9B0BFE9AC76F/__I23_data__/13284/regiongrow/"
imagename = "SEGMENTED_4phases_13284_processed_processed_processed_processed_processed_"

tot_num_im = 1000
volume_automatic_regiongrow = np.zeros((tot_num_im, 395, 755)).astype(np.uint8)

for i in range(0,tot_num_im):
    if (i < 10):
        index_str = '0000' + str(i)
    elif (10 <= i < 100):
        index_str = '000' + str(i)
    elif (100 <= i < 1000):
        index_str = '00' + str(i)
    else:
        index_str = str(i)
    volume_automatic_regiongrow[i,:,:] = np.array(Image.open(path_folder + imagename + index_str + ".tif"))

#%%
# change classes to fit manual segmentation:
volume_automatic_regiongrow2 = np.zeros((395, 1000, 755)).astype(np.uint8)
volume_automatic_regiongrow[volume_automatic_regiongrow == 5] = 0 # vacuum
volume_automatic_regiongrow[volume_automatic_regiongrow == 200] = 1 # crystal
volume_automatic_regiongrow[volume_automatic_regiongrow == 255] = 3 # liquor
volume_automatic_regiongrow[volume_automatic_regiongrow == 186] = 2 # loop
volume_automatic_regiongrow[0:28,:,:] = 0 

for i in range(0,755):
    volume_automatic_regiongrow2[:,:,i] = np.transpose(volume_automatic_regiongrow[:,:,i])

volume_automatic_regiongrow = volume_automatic_regiongrow2
del volume_automatic_regiongrow2
#%%
fig = plt.figure(3)
#fig.suptitle('Automatic 4 phases segmentation GMM + Regiongrow (crystal)')
ax = fig.add_subplot(131)
plt.imshow(volume_automatic_regiongrow[:,550,:],vmin=0, vmax=3)
ax.set_aspect(1.3)
plt.axis('off')
#plt.title('axial view')

ax = fig.add_subplot(132)
plt.imshow(volume_automatic_regiongrow[160,:,:],vmin=0, vmax=3)
ax.set_aspect(0.75)
plt.axis('off')
#plt.title('coronal view')

ax = fig.add_subplot(133)
plt.imshow(volume_automatic_regiongrow[:,:,230],vmin=0, vmax=3)
plt.axis('off')
ax.set_aspect(1.5)
#plt.title('sagittal view')
fig.set_size_inches((13, 8), forward=False)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()
fig.savefig('13284_regiongrow.png', dpi=250)
#%%

#%%
# Compare the result of segmentation (quantitative approach for each class separetly)
class_to_calculate_error = 1 # identify the class of interest
segm_class_man = np.zeros(np.shape(volume_manual)).astype(np.int8)
segm_class_auto = np.zeros(np.shape(volume_manual)).astype(np.int8)
segm_class_man[volume_manual == class_to_calculate_error] = 1
segm_class_auto[volume_automatic_regiongrow == class_to_calculate_error] = 1

segm_diff = abs(segm_class_man - segm_class_auto)

fig = plt.figure(4)
#fig.suptitle('Error between manual and Region segmentation (crystal)')

ax = fig.add_subplot(131)
plt.imshow(segm_diff[:,550,:], vmin=0, vmax=1)
plt.axis('off')
#plt.title('segmentation error, axial view')

ax = fig.add_subplot(132)
plt.imshow(segm_diff[160,:,:], vmin=0, vmax=1)
plt.axis('off')
#plt.title('segmentation error, coronal view')

ax = fig.add_subplot(133)
plt.imshow(segm_diff[:,:,230], vmin=0, vmax=1)
plt.axis('off')
#plt.title('segmentation error, sagittal view')
fig.set_size_inches((13, 8), forward=False)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()
fig.savefig('13284_regiongrow_crystal_error.png', dpi=250)

# plotting the error for each slice
vector_manual = np.zeros((tot_num_im)).astype(np.int64)
vector_automatic_regiongrow = np.zeros((tot_num_im)).astype(np.int64)
vector_errors = np.zeros((tot_num_im)).astype(np.int64)
for i in range(0,tot_num_im):
    vector_errors[i] = np.count_nonzero(segm_diff[:,i,:])
    vector_manual[i] = np.count_nonzero(segm_class_man[:,i,:])
    vector_automatic_regiongrow[i] = np.count_nonzero(segm_class_auto[:,i,:])
#%%
# visualise the result
ind = np.arange(tot_num_im)
barWidth = 1.5
names = ['Manual','RegionGrow']

fig = plt.figure(10)
plt.rcParams.update({'font.size': 15})
p1 = plt.bar(ind, vector_manual,  alpha = 0.7, color='r', width=barWidth)
p2 = plt.bar(ind, vector_automatic_regiongrow, alpha = 0.25, color='b', width=barWidth)
plt.xlim([300,800])
#plt.yticks(fontsize=12)
plt.ylabel('The total number of pixels', fontsize=14)
plt.xlabel('2D (x-y) slices', fontsize=14)
plt.legend((p1[0], p2[0]), (names[0], names[1]), fontsize=16, ncol=2, framealpha=0, fancybox=True)
plt.title('RegionGrow vs Manual for crystal segementation')
plt.show()
#fig = plt.gcf()
fig.set_size_inches((9, 11), forward=False)
fig.savefig('13284_crystal_regiongrow.png', dpi=250)
#%%

# Getting automatically segmented data
path_folder = "/media/algol/F2FE9B0BFE9AC76F/__I23_data__/13284/geodistance/"
imagename = "SEGMENTED_4phases_13284_processed_processed_processed_processed_processed_"

tot_num_im = 1000
volume_automatic_geodist= np.zeros((tot_num_im, 395, 755)).astype(np.uint8)

for i in range(0,tot_num_im):
    if (i < 10):
        index_str = '0000' + str(i)
    elif (10 <= i < 100):
        index_str = '000' + str(i)
    elif (100 <= i < 1000):
        index_str = '00' + str(i)
    else:
        index_str = str(i)
    volume_automatic_geodist[i,:,:] = np.array(Image.open(path_folder + imagename + index_str + ".tif"))


#%%
# change classes to fit manual segmentation:
volume_automatic_geodist2 = np.zeros((395, 1000, 755)).astype(np.uint8)
volume_automatic_geodist[volume_automatic_geodist == 5] = 0 # vacuum
volume_automatic_geodist[volume_automatic_geodist == 200] = 1 # crystal
volume_automatic_geodist[volume_automatic_geodist == 255] = 3 # liquor
volume_automatic_geodist[volume_automatic_geodist == 186] = 2 # loop
volume_automatic_geodist[0:28,:,:] = 0 

for i in range(0,755):
    volume_automatic_geodist2[:,:,i] = np.transpose(volume_automatic_geodist[:,:,i])

volume_automatic_geodist = volume_automatic_geodist2
del volume_automatic_geodist2

#%%
fig = plt.figure(5)
#fig.suptitle('Automatic 4 phases segmentation GMM + Geodist (crystal)')
ax = fig.add_subplot(131)
plt.imshow(volume_automatic_geodist[:,550,:],vmin=0, vmax=3)
ax.set_aspect(1.3)
plt.axis('off')
#plt.title('axial view')

ax = fig.add_subplot(132)
plt.imshow(volume_automatic_geodist[160,:,:],vmin=0, vmax=3)
ax.set_aspect(0.75)
plt.axis('off')
#plt.title('coronal view')

ax = fig.add_subplot(133)
plt.imshow(volume_automatic_geodist[:,:,230],vmin=0, vmax=3)
plt.axis('off')
ax.set_aspect(1.5)
#plt.title('sagittal view')
fig.set_size_inches((13, 8), forward=False)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()
fig.savefig('13284_geodist.png', dpi=250)

#%%
# Compare the result of segmentation (quantitative approach for each class separetly)
class_to_calculate_error = 1 # identify the class of interest
segm_class_man = np.zeros(np.shape(volume_manual)).astype(np.int8)
segm_class_auto = np.zeros(np.shape(volume_manual)).astype(np.int8)
segm_class_man[volume_manual == class_to_calculate_error] = 1
segm_class_auto[volume_automatic_geodist == class_to_calculate_error] = 1

segm_diff = abs(segm_class_man - segm_class_auto)

fig = plt.figure(4)
#fig.suptitle('Error between manual and Region segmentation (crystal)')

ax = fig.add_subplot(131)
plt.imshow(segm_diff[:,550,:], vmin=0, vmax=1)
plt.axis('off')
#plt.title('segmentation error, axial view')

ax = fig.add_subplot(132)
plt.imshow(segm_diff[160,:,:], vmin=0, vmax=1)
plt.axis('off')
#plt.title('segmentation error, coronal view')

ax = fig.add_subplot(133)
plt.imshow(segm_diff[:,:,230], vmin=0, vmax=1)
plt.axis('off')
#plt.title('segmentation error, sagittal view')
fig.set_size_inches((13, 8), forward=False)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()
fig.savefig('13284_geodist_crystal_error.png', dpi=250)

# plotting the error for each slice
vector_manual = np.zeros((tot_num_im)).astype(np.int64)
vector_automatic_geodist = np.zeros((tot_num_im)).astype(np.int64)
vector_errors = np.zeros((tot_num_im)).astype(np.int64)
for i in range(0,tot_num_im):
    vector_errors[i] = np.count_nonzero(segm_diff[:,i,:])
    vector_manual[i] = np.count_nonzero(segm_class_man[:,i,:])
    vector_automatic_geodist[i] = np.count_nonzero(segm_class_auto[:,i,:])
#%%

ind = np.arange(tot_num_im)
barWidth = 1.5
names = ['Manual','GeoDistance']

fig = plt.figure(10)
plt.rcParams.update({'font.size': 15})
p1 = plt.bar(ind, vector_manual,  alpha = 0.7, color='r', width=barWidth)
p2 = plt.bar(ind, vector_automatic_geodist, alpha = 0.25, color='b', width=barWidth)
plt.xlim([300,800])
#plt.yticks(fontsize=12)
plt.ylabel('The total number of pixels', fontsize=14)
plt.xlabel('2D (x-y) slices', fontsize=14)
plt.legend((p1[0], p2[0]), (names[0], names[1]), fontsize=16, ncol=2, framealpha=0, fancybox=True)
plt.title('GeoDistance vs Manual for crystal segementation')
plt.show()
#fig = plt.gcf()
fig.set_size_inches((9, 11), forward=False)
fig.savefig('13284_crystal_geodist.png', dpi=250)
#%%
names = ['Geodistance','Regiongrow']
fig = plt.figure(11)
plt.rcParams.update({'font.size': 15})
p1 = plt.plot(vector_manual-vector_automatic_geodist, linewidth=2.0, linestyle='-.')
p2 = plt.plot(vector_manual-vector_automatic_regiongrow, linewidth=2.0)
plt.axhline(linewidth=1, linestyle='--', color='black')
plt.xlim([300,800])
#plt.yticks(fontsize=12)
plt.ylabel('Pixel differences', fontsize=14)
plt.xlabel('2D (x-y) slices', fontsize=14)
plt.legend((p1[0], p2[0]), (names[0], names[1]), fontsize=16, ncol=2, framealpha=0, fancybox=True)
plt.title('Geodistance and Regiongrow deviation from manual segmentation (crystal)')
plt.show()
#fig = plt.gcf()
fig.set_size_inches((11, 7), forward=False)
fig.savefig('13284_geodist_vs_regiongrow.png', dpi=250)
