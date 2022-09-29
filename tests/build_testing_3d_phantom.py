import numpy as np
import timeit
import bz2
import IPython
import matplotlib
import matplotlib.pyplot as plt

from larix.methods.misc import MEDIAN_FILT
from larix.methods.misc_gpu import MEDIAN_FILT_GPU

import tomophantom
from tomophantom import TomoP3D
import os

model = 13 # select a model
N_size = 1024 
# one can specify an exact path to the parameters file
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")
print("Creating a phantom...")
phantom_3D = TomoP3D.Model(model, N_size, path_library3D)

print("Applying noise to the phantom...")
phantom_3D += np.random.normal(loc = 0.0, scale = 0.2, size = np.shape(phantom_3D))

print("Applying Median Filter in 3D using the CPU...")
pars = {'input_data' : np.float32(phantom_3D), # input grayscale image
        'kernel_size' : 3}

start_time = timeit.default_timer()
volume_filteredCPU = MEDIAN_FILT(pars['input_data'], pars['kernel_size'])
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

pars = {'input_data' : np.float32(phantom_3D), # input grayscale image
        'kernel_half_size' : 1}

print("Applying Median Filter in 3D using the GPU...")
start_time = timeit.default_timer()
volume_filteredGPU = MEDIAN_FILT_GPU(pars['input_data'], pars['kernel_half_size'])
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

print("The sum of the CPU and GPU residual must be zero and  it equals to:")
print(np.sum(volume_filteredCPU - volume_filteredGPU))

midval = (int)(N_size/2)
matplotlib.image.imsave('noisy_image.png', phantom_3D[:,midval,:])
matplotlib.image.imsave('denoisedCPU.png', volume_filteredCPU[:,midval,:])
matplotlib.image.imsave('denoisedGPU.png', volume_filteredGPU[:,midval,:])
matplotlib.image.imsave('denoised_CPU_GPU_residual.png', np.abs(volume_filteredCPU[:,midval,:] - volume_filteredGPU[:,midval,:]))
#IPython.embed()

