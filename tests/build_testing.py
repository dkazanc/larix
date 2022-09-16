# a test script to run automatically after the build

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import timeit
from numpy import random

from larix.methods.misc import MEDIAN_FILT, MEDIAN_DEZING
from larix.methods.misc_gpu import MEDIAN_FILT_GPU, MEDIAN_DEZING_GPU

#image2d = np.float32(random.random((150,250)))   # Test data
image2d = np.load('sino_noisy.npy') # load noisy sinogram
image2d_filteredCPU = np.load('sino_denoiseCPU.npy') # the CPU benchmark

print("Applying Median Filter in 2D using GPU...")

pars = {'input_data' : image2d, # input a grayscale image
        'kernel_size' : 3}

start_time = timeit.default_timer()
image2d_filteredGPU = MEDIAN_FILT_GPU(pars['input_data'], pars['kernel_size'])
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

print("The sum of the CPU and GPU residual must be zero, it equals to:")
print(np.sum(image2d_filteredGPU - image2d_filteredCPU))

matplotlib.image.imsave('noisy_image.png', image2d)
matplotlib.image.imsave('denoisedCPU.png', image2d_filteredCPU)
matplotlib.image.imsave('denoisedGPU.png', image2d_filteredGPU)
matplotlib.image.imsave('denoised_CPU_GPU_residual.png', np.abs(image2d_filteredGPU - image2d_filteredCPU))
