#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created April 2020

Demo to show the capability of the dezinger and median filter 
You will need to install TomoPhantom for this demo:
conda install -c dkazanc tomophantom
@author: Daniil Kazantsev
"""
import numpy as np
import matplotlib.pyplot as plt

from larix.methods.misc import MEDIAN_FILT, MEDIAN_DEZING
from larix.methods.misc_gpu import MEDIAN_FILT_GPU, MEDIAN_DEZING_GPU

import timeit
from tomophantom import TomoP2D
import os
import tomophantom
from tomophantom.supp.artifacts import _Artifacts_

model = 13 # select a model
N_size = 2000 # set dimension of the phantom
# one can specify an exact path to the parameters file
# path_library2D = '../../../PhantomLibrary/models/Phantom2DLibrary.dat'
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
phantom_2D = TomoP2D.Model(model, N_size, path_library2D)

# get analytical sinogram
# parameters to generate a sinogram
angles_num = int(0.5*np.pi*N_size); # angles number
angles = np.linspace(0.0,179.9,angles_num,dtype='float32')
angles_rad = angles*(np.pi/180.0)
P = int(np.sqrt(2)*N_size) #detectors
sino_an = TomoP2D.ModelSino(model, N_size, P, angles, path_library2D)

# forming dictionaries with artifact types
_noise_ =  {'noise_type' : 'Gaussian',
            'noise_sigma' : 3.5, # noise amplitude
            'noise_seed' : 0}

# adding zingers
_zingers_ = {'zingers_percentage' : 0.3,
             'zingers_modulus' : 15}

sino_an_noisy = _Artifacts_(sino_an, **_noise_, **_zingers_)

plt.figure(1)
plt.rcParams.update({'font.size': 21})
plt.imshow(sino_an_noisy, cmap="BuPu")
plt.title('{}''{}'.format('Analytical sinogram of model no.',model))

print("Applying Median Filter in 2D...")

pars = {'input_data' : sino_an_noisy, # input grayscale image
        'kernel_size' : 5}

start_time = timeit.default_timer()
filtered = MEDIAN_FILT(pars['input_data'], pars['kernel_size'])
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

plt.figure(2)
plt.rcParams.update({'font.size': 21})
plt.imshow(filtered, cmap="BuPu")
plt.title('{}''{}'.format('Filtered sinogram of model no.',model))

print("Applying Dezinger Filter in 2D...")

pars = {'input_data' : sino_an_noisy, # input grayscale image
        'kernel_size' : 5,
        'mu_threshold': 100.0}

start_time = timeit.default_timer()
dezingered = MEDIAN_DEZING(pars['input_data'], pars['kernel_size'], pars['mu_threshold'])
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

plt.figure(3)
plt.rcParams.update({'font.size': 21})
plt.imshow(dezingered, cmap="BuPu")
plt.title('{}''{}'.format('Dezingered sinogram of model no.',model))

#%%
print("Applying Median Filter in 2D using GPU...")

pars = {'input_data' : sino_an_noisy, # input grayscale image
        'kernel_size' : 5}

start_time = timeit.default_timer()
filtered_gpu = MEDIAN_FILT_GPU(pars['input_data'], pars['kernel_size'])
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

plt.figure(4)
plt.rcParams.update({'font.size': 21})
plt.imshow(filtered_gpu, cmap="BuPu")
plt.title('{}''{}'.format('GPU Filtered sinogram of model no.',model))

print("Applying GPU-accelerated Dezinger Filter in 2D...")

pars = {'input_data' : sino_an_noisy, # input grayscale image
        'kernel_size' : 5,
        'mu_threshold': 100.0}

start_time = timeit.default_timer()
gpu_dezingered = MEDIAN_DEZING_GPU(pars['input_data'], pars['kernel_size'], pars['mu_threshold'])
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

plt.figure(5)
plt.rcParams.update({'font.size': 21})
plt.imshow(gpu_dezingered, cmap="BuPu")
plt.title('{}''{}'.format('GPU Dezingered sinogram of model no.',model))
