# a test script to run automatically after the build

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import timeit
from numpy import random

from larix.methods.misc import MEDIAN_FILT, MEDIAN_DEZING
#from larix.methods.misc_gpu import MEDIAN_FILT_GPU, MEDIAN_DEZING_GPU
from larix.methods.misc_gpu import MEDIAN_FILT_GPU, MEDIAN_FILT_GPU_SHARED

image2d = np.float32(random.random((150,250)))   # Test data
#image2d = np.load('sino_noisy.npy') # load noisy sinogram
#image2d_filteredCPU = np.load('sino_denoiseCPU.npy') # the CPU benchmark

pars = {'input_data' : np.float32(image2d), # input a grayscale image
        'radius' : 1}

print("Applying Median Filter in 2D using the CPU...")
pars = {'input_data' : np.float32(image2d), # input a grayscale image
        'radius' : 1}
image2d_filteredCPU = MEDIAN_FILT(pars['input_data'], pars['radius'])

print("Applying Median Filter in 2D using the GPU...")

start_time = timeit.default_timer()
image2d_filteredGPU = MEDIAN_FILT_GPU(pars['input_data'], pars['radius'])
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

print("The sum of the CPU and GPU residual must be zero, it equals to:")
print(np.sum(image2d_filteredGPU - image2d_filteredCPU))

matplotlib.image.imsave('noisy_image.png', image2d)
matplotlib.image.imsave('denoisedCPU.png', image2d_filteredCPU)
matplotlib.image.imsave('denoisedGPU.png', image2d_filteredGPU)
matplotlib.image.imsave('denoised_CPU_GPU_residual.png', np.abs(image2d_filteredGPU - image2d_filteredCPU))
