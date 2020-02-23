#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:39:47 2020

Demo to show the capability of segmenting the phase of the 3D data with the 
subsequent morphological processing 
-----------------------------------------------------
Download the data from the Zenodo archive here:
https://doi.org/10.5281/zenodo.3685282

Place the downloaded file sample13076_3D.npy into the "data" folder
-----------------------------------------------------
@author: Daniil Kazantsev 
"""
import numpy as np
import matplotlib.pyplot as plt
from colleda.methods.segmentation import MASK_EVOLVE, MASK_MORPH

#  Load the 3D sample data (i23 beamline, DLS)
sample_data =  np.load('../data/sample13076_3D.npy')

mask_init = np.uint8(np.zeros(np.shape(sample_data)))
mask_init[20:100,325:414,230:317] = 1

selected_slice_vis = 30
fig= plt.figure()
plt.rcParams.update({'font.size': 21})
plt.subplot(121)
plt.imshow(sample_data[selected_slice_vis,:,:], vmin=0, vmax=0.5, cmap="gray")
plt.title('Original Image')
plt.subplot(122)
plt.imshow(mask_init[selected_slice_vis,:,:], vmin=0, vmax=1, cmap="gray")
plt.title('Phase specific initialised mask')
plt.show()
#%%
print("Runnning mask evolving segmentation in 3D...")

pars = {'input_data' : sample_data, # input grayscale volume
        'maskData' : mask_init,     # generated initialisation mask
        'threhsold' : 9.8 ,         # threhsold controls where evolution stops (>=1)
        'iterationsNumb' : 250,     # the number of iterations (depends on the size of the phase)
        'connectivity' : 6,         # voxel connectivity rule, choose between 4 (2D), 6, 8 (2D), and 26
        'method' : 'mean'}          # method to collect statistics from the mask (mean. median, value)

mask_evolved = MASK_EVOLVE(pars['input_data'], pars['maskData'],\
                           pars['threhsold'], pars['iterationsNumb'],\
                           pars['connectivity'], pars['method'])

fig= plt.figure()
plt.rcParams.update({'font.size': 21})
plt.subplot(121)
plt.imshow(sample_data[selected_slice_vis,:,:], vmin=0, vmax=0.5, cmap="gray")
plt.title('Original Volume')
plt.subplot(122)
plt.imshow(mask_evolved[selected_slice_vis,:,:], vmin=0, vmax=1, cmap="gray")
plt.title('Evolution of the 3D mask')
plt.show()
#%%
print("Morphological processing the resulting mask in 3D (will take some time)...")

pars = {'maskdata' : mask_evolved,# input binary mask
        'primeClass' : 0,         # class to start morphological processing from        
        'CorrectionWindow' : 7 ,  # the non-local neighboorhood window 
        'iterationsNumb' : 3}     # iterations number (less for 3D than 2D)

mask_morphed = MASK_MORPH(pars['maskdata'], pars['primeClass'], 
                          pars['CorrectionWindow'], pars['iterationsNumb'])

fig= plt.figure()
plt.rcParams.update({'font.size': 21})
plt.subplot(121)
plt.imshow(mask_evolved[selected_slice_vis,:,:], vmin=0, vmax=1, cmap="gray")
plt.title('Evolved mask')
plt.subplot(122)
plt.imshow(mask_morphed[selected_slice_vis,:,:], vmin=0, vmax=1, cmap="gray")
plt.title('Processed 3D mask')
plt.show()
#%%
##################Getting a different phase ###################
mask_init = np.uint8(np.zeros(np.shape(sample_data)))
mask_init[50:100,653:774,147:283] = 1
mask_init[60:70,135:171,165:199] = 1

print("Runnning mask evolving segmentation in 3D...")

pars = {'input_data' : sample_data, # input grayscale volume
        'maskData' : mask_init,     # generated initialisation mask
        'threhsold' : 9.0 ,         # threhsold controls where evolution stops (>=1)
        'iterationsNumb' : 250,     # the number of iterations (depends on the size of the phase)
        'connectivity' : 6,         # voxel connectivity rule, choose between 4 (2D), 6, 8 (2D), and 26
        'method' : 'mean'}          # method to collect statistics from the mask (mean. median, value)

mask_evolved = MASK_EVOLVE(pars['input_data'], pars['maskData'],\
                           pars['threhsold'], pars['iterationsNumb'],\
                           pars['connectivity'], pars['method'])

fig= plt.figure()
plt.rcParams.update({'font.size': 21})
plt.subplot(121)
plt.imshow(sample_data[selected_slice_vis,:,:], vmin=0, vmax=0.5, cmap="gray")
plt.title('Original Volume')
plt.subplot(122)
plt.imshow(mask_evolved[selected_slice_vis,:,:], vmin=0, vmax=1, cmap="gray")
plt.title('Evolution of the 3D mask')
plt.show()
#%%
print("Morphological processing of the resulting mask in 3D (will take some time)...")

pars = {'maskdata' : mask_evolved, # input binary mask
        'primeClass' : 1,          # class to start morphological processing from        
        'CorrectionWindow' : 6,    # the non-local neighboorhood window 
        'iterationsNumb' : 2}      # iterations number (less for 3D than 2D)

mask_morphed = MASK_MORPH(pars['maskdata'], pars['primeClass'], 
                          pars['CorrectionWindow'], pars['iterationsNumb'])

fig= plt.figure()
plt.rcParams.update({'font.size': 21})
plt.subplot(121)
plt.imshow(mask_evolved[selected_slice_vis,:,:], vmin=0, vmax=1, cmap="gray")
plt.title('Evolved mask')
plt.subplot(122)
plt.imshow(mask_morphed[selected_slice_vis,:,:], vmin=0, vmax=1, cmap="gray")
plt.title('Processed 3D mask')
plt.show()
#%%
# save images and superimpose
from PIL import Image
import matplotlib

matplotlib.image.imsave('background3d.png', sample_data[selected_slice_vis,:,:])
matplotlib.image.imsave('foreground3d.png', mask_morphed[selected_slice_vis,:,:])

background = Image.open("background3d.png")
overlay = Image.open("foreground3d.png")

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.4)
new_img.save("overlay3d.png","PNG")
#%%