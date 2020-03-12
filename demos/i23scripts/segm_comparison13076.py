#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thsi script calculates an error between two segmentations
@author: Daniil Kazantsev
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

tot_num_im = 900
# read reconstructed images
path_folder = "/dls/i23/data/2019/nr23017-1/processing/tomography/recon/13076/PAPER_PROC/tiffs/cropped/"
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
fig.suptitle('TomoBAR reconstructed image')
plt.subplot(131)
plt.imshow(volume_recon[450,:,:],vmin=0, vmax=255,  cmap="gray")
plt.title('axial view')

plt.subplot(132)
plt.imshow(np.flipud(volume_recon[:,250,:]),vmin=0, vmax=255, cmap="gray")
plt.title('coronal view')

plt.subplot(133)
plt.imshow(np.flipud(volume_recon[:,:,200]),vmin=0, vmax=255,  cmap="gray")
plt.title('sagittal view')
fig.set_size_inches((13, 8), forward=False)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()
fig.savefig('13076_tomobar_recon.png', dpi=250)

#%%
path_folder = "/dls/i23/data/2019/nr23017-1/processing/tomography/recon/13076/PAPER_PROC/avizo/tiffs_cropped_xy_slicing/"
imagename = "13076_copped_thresholding_magic_wand.labels"

tot_num_im = 900
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
plt.subplot(131)
plt.imshow(volume_manual[450,:,:],vmin=0, vmax=3)
plt.axis('off')
#plt.title('axial view')

plt.subplot(132)
plt.imshow(np.flipud(volume_manual[:,250,:]),vmin=0, vmax=3)
plt.axis('off')
#plt.title('coronal view')

plt.subplot(133)
plt.imshow(np.flipud(volume_manual[:,:,200]),vmin=0, vmax=3)
plt.axis('off')
#plt.title('sagittal view')
fig.set_size_inches((13, 8), forward=False)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()
fig.savefig('13076_manual.png', dpi=250)
#%%
# Getting automatically segmented data
path_folder = "/dls/i23/data/2019/nr23017-1/processing/tomography/recon/13076/PAPER_PROC/MaskEvolve3D/FINAL_SEGMENTATION/ImageSaver-SEGMENTED_4phases/"
imagename = "SEGMENTED_4phases_13076_processed_processed_processed_processed_processed_"

volume_automatic_regiongrow = np.zeros((tot_num_im, dimX, dimY)).astype(np.uint8)

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

# change classes to fit manual segmentation:
volume_automatic_regiongrow[volume_automatic_regiongrow == 255] = 3 # liquor
volume_automatic_regiongrow[volume_automatic_regiongrow == 7] = 2 # loop
volume_automatic_regiongrow[volume_automatic_regiongrow == 198] = 1 # crystal

fig = plt.figure(3)
#fig.suptitle('Automatic 4 phases segmentation GMM + Regiongrow (crystal)')

plt.subplot(131)
plt.imshow(volume_automatic_regiongrow[450,:,:],vmin=0, vmax=3)
plt.axis('off')
#plt.title('axial view')

plt.subplot(132)
plt.imshow(np.flipud(volume_automatic_regiongrow[:,250,:]),vmin=0, vmax=3)
plt.axis('off')
#plt.title('coronal view')

plt.subplot(133)
plt.imshow(np.flipud(volume_automatic_regiongrow[:,:,200]),vmin=0, vmax=3)
plt.axis('off')
#plt.title('sagittal view')
fig.set_size_inches((13, 8), forward=False)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()
fig.savefig('13076_regiongrow.png', dpi=250)
#%%
# Compare the result of segmentation (quantitative approach for each class separetly)
class_to_calculate_error = 1 # identify the class of interest
segm_class_man = np.zeros((tot_num_im, dimX, dimY)).astype(np.int8)
segm_class_auto = np.zeros((tot_num_im, dimX, dimY)).astype(np.int8)
segm_class_man[volume_manual == class_to_calculate_error] = 1
segm_class_auto[volume_automatic_regiongrow == class_to_calculate_error] = 1

segm_diff = abs(segm_class_man - segm_class_auto)

fig = plt.figure(4)
#fig.suptitle('Error between manual and Region segmentation (crystal)')

plt.subplot(311)
plt.imshow(segm_diff[450,:,:], vmin=0, vmax=1)
plt.axis('off')
#plt.title('segmentation error, axial view')

plt.subplot(312)
plt.imshow(np.flipud(segm_diff[:,250,:]), vmin=0, vmax=1)
plt.axis('off')
#plt.title('segmentation error, coronal view')

plt.subplot(313)
plt.imshow(np.flipud(segm_diff[:,:,200]), vmin=0, vmax=1)
plt.axis('off')
#plt.title('segmentation error, sagittal view')
fig.set_size_inches((13, 8), forward=False)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()
fig.savefig('13076_regiongrow_error.png', dpi=250)

# plotting the error for each slice
vector_manual = np.zeros((tot_num_im)).astype(np.int64)
vector_automatic_regiongrow = np.zeros((tot_num_im)).astype(np.int64)
vector_errors = np.zeros((tot_num_im)).astype(np.int64)
for i in range(0,tot_num_im):
    vector_errors[i] = np.count_nonzero(segm_diff[i,:,:])
    vector_manual[i] = np.count_nonzero(segm_class_man[i,:,:])
    vector_automatic_regiongrow[i] = np.count_nonzero(segm_class_auto[i,:,:])
#%%
# visualise the result    
ind = np.arange(tot_num_im) 
barWidth = 1.5
names = ['Manual','Automatic']

fig = plt.figure(10)
plt.rcParams.update({'font.size': 15})
p1 = plt.bar(ind, vector_manual,  alpha = 0.7, color='r', width=barWidth)
p2 = plt.bar(ind, vector_automatic_regiongrow, alpha = 0.25, color='b', width=barWidth)
plt.xlim([0,900])
#plt.yticks(fontsize=12)
plt.ylabel('The total number of pixels', fontsize=14)
plt.xlabel('2D (x-y) slices', fontsize=14)
plt.legend((p1[0], p2[0]), (names[0], names[1]), fontsize=16, ncol=2, framealpha=0, fancybox=True)
plt.title('Regiongrow and manual segementation discrepancies for a crystal')
plt.show()
#fig = plt.gcf()
fig.set_size_inches((9, 11), forward=False)
fig.savefig('13076_CRYSTAL_regiongrow.png', dpi=250)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Getting automatically segmented data
path_folder = "/dls/i23/data/2019/nr23017-1/processing/tomography/recon/13076/PAPER_PROC/GeoDist3D/FINAL_SEGMENTATION/ImageSaver-SEGMENTED_4phases/"
imagename = "SEGMENTED_4phases_13076_processed_processed_processed_processed_processed_"

volume_automatic_geodistance = np.zeros((tot_num_im, dimX, dimY)).astype(np.uint8)

for i in range(0,tot_num_im):
    if (i < 10):
        index_str = '0000' + str(i)
    elif (10 <= i < 100):
        index_str = '000' + str(i)
    elif (100 <= i < 1000):
        index_str = '00' + str(i)
    else:
        index_str = str(i)
    volume_automatic_geodistance[i,:,:] = np.array(Image.open(path_folder + imagename + index_str + ".tif"))

# change classes to fit manual segmentation:
volume_automatic_geodistance[volume_automatic_geodistance == 255] = 3 # liquor
volume_automatic_geodistance[volume_automatic_geodistance == 7] = 2 # loop
volume_automatic_geodistance[volume_automatic_geodistance == 198] = 1 # crystal

fig = plt.figure(3)
#fig.suptitle('Automatic 4 phases segmentation GMM + Regiongrow (crystal)')

plt.subplot(131)
plt.imshow(volume_automatic_geodistance[450,:,:],vmin=0, vmax=3)
plt.axis('off')
#plt.title('axial view')

plt.subplot(132)
plt.imshow(np.flipud(volume_automatic_geodistance[:,250,:]),vmin=0, vmax=3)
plt.axis('off')
#plt.title('coronal view')

plt.subplot(133)
plt.imshow(np.flipud(volume_automatic_geodistance[:,:,200]),vmin=0, vmax=3)
plt.axis('off')
#plt.title('sagittal view')
fig.set_size_inches((13, 8), forward=False)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()
fig.savefig('13076_geodistance.png', dpi=250)
#%%
# Compare the result of segmentation (quantitative approach for each class separetly)
class_to_calculate_error = 1 # identify the class of interest
segm_class_man = np.zeros((tot_num_im, dimX, dimY)).astype(np.int8)
segm_class_auto = np.zeros((tot_num_im, dimX, dimY)).astype(np.int8)
segm_class_man[volume_manual == class_to_calculate_error] = 1
segm_class_auto[volume_automatic_geodistance == class_to_calculate_error] = 1

segm_diff = abs(segm_class_man - segm_class_auto)

fig = plt.figure(4)
#fig.suptitle('Error between manual and Geodistance segmentation (crystal)')

plt.subplot(311)
plt.imshow(segm_diff[450,:,:], vmin=0, vmax=1)
plt.axis('off')
#plt.title('segmentation error, axial view')

plt.subplot(312)
plt.imshow(np.flipud(segm_diff[:,250,:]), vmin=0, vmax=1)
plt.axis('off')
#plt.title('segmentation error, coronal view')

plt.subplot(313)
plt.imshow(np.flipud(segm_diff[:,:,200]), vmin=0, vmax=1)
plt.axis('off')
#plt.title('segmentation error, sagittal view')
fig.set_size_inches((13, 8), forward=False)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()
fig.savefig('13076_geodistance_error.png', dpi=250)

# plotting the error for each slice
vector_manual = np.zeros((tot_num_im)).astype(np.int64)
vector_automatic_geodistance = np.zeros((tot_num_im)).astype(np.int64)
vector_errors = np.zeros((tot_num_im)).astype(np.int64)
for i in range(0,tot_num_im):
    vector_errors[i] = np.count_nonzero(segm_diff[i,:,:])
    vector_manual[i] = np.count_nonzero(segm_class_man[i,:,:])
    vector_automatic_geodistance[i] = np.count_nonzero(segm_class_auto[i,:,:])
#%%
# visualise the result    
ind = np.arange(tot_num_im) 
barWidth = 1.5
names = ['Manual','Automatic']

fig = plt.figure(10)
plt.rcParams.update({'font.size': 15})
p1 = plt.bar(ind, vector_manual,  alpha = 0.7, color='r', width=barWidth)
p2 = plt.bar(ind, vector_automatic_geodistance, alpha = 0.25, color='b', width=barWidth)
plt.xlim([0,900])
#plt.yticks(fontsize=12)
plt.ylabel('The total number of pixels', fontsize=14)
plt.xlabel('2D (x-y) slices', fontsize=14)
plt.legend((p1[0], p2[0]), (names[0], names[1]), fontsize=16, ncol=2, framealpha=0, fancybox=True)
plt.title('Regiongrow and manual segementation discrepancies for a crystal')
plt.show()
#fig = plt.gcf()
fig.set_size_inches((9, 11), forward=False)
fig.savefig('13076_CRYSTAL_geodistance.png', dpi=250)

#%%
names = ['Geodistance','Regiongrow']
fig = plt.figure(11)
plt.rcParams.update({'font.size': 15})
p1 = plt.plot(vector_manual-vector_automatic_geodistance)
p2 = plt.plot(vector_manual-vector_automatic_regiongrow)
#plt.xlim([0,900])
#plt.yticks(fontsize=12)
plt.ylabel('Pixel differences', fontsize=14)
plt.xlabel('2D (x-y) slices', fontsize=14)
plt.legend((p1[0], p2[0]), (names[0], names[1]), fontsize=16, ncol=2, framealpha=0, fancybox=True)
plt.title('Geodistance and Regiongrow deviation from manual segmentation (crystal)')
plt.show()
#fig = plt.gcf()
fig.set_size_inches((11, 7), forward=False)
fig.savefig('13076_geodist_vs_regiongrow.png', dpi=250)
#%%
# save images and superimpose
from PIL import Image
import matplotlib

matplotlib.image.imsave('background.png', np.flipud(volume_recon[:,250,:]))
matplotlib.image.imsave('foreground.png', np.flipud(volume_manual[:,250,:]))

background = Image.open("background.png")
overlay = Image.open("foreground.png")

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.3)
new_img.save("new.png","PNG")
#%%