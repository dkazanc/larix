#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 2020

Demo to show the capability of autocropping function

@author: Daniil Kazantsev 
"""
import numpy as np
import matplotlib.pyplot as plt
from larix.methods.misc import AUTOCROP

#  Load the 2D sample data (i23 beamline, DLS)
#sample_data = np.load('../data/data2D_to_crop.npy')
sample_data = np.load('/home/kjy41806/Documents/SOFT/larix/data/data2D_to_crop.npy')


plt.figure(1)
plt.imshow(sample_data, vmin=0, vmax=2, cmap="gray")
plt.title('2D tomographic projection')
plt.show()

#%%
print("Runnning autocropping in 2D...")

pars = {'input_data' : sample_data, # input grayscale image
        'threshold' : 0.08,
        'margin_skip' : 10,         # 
        'statbox_size' : 20,
        'increase_crop' : 50}        # method to collect statistics from the mask (mean. median, value)

cropped_indices = AUTOCROP(pars['input_data'], pars['threshold'],\
                           pars['margin_skip'], pars['statbox_size'],\
                           pars['increase_crop'])

"""
plt.figure(2)
plt.imshow(mask_evolved, vmin=0, vmax=0.8, cmap="gray")
plt.title('2D result')
plt.show()
"""
#%%