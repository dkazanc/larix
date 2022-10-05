import numpy as np
import timeit
import bz2
import IPython
import matplotlib
import matplotlib.pyplot as plt

from larix.methods.misc import MEDIAN_FILT
from larix.methods.misc_gpu import MEDIAN_FILT_GPU, MEDIAN_FILT_GPU_SHARED

size_tuple = (50,50,100)
#size_tuple = (500,500,00)
print("Creating a 3D array...")
Noise3DArray = np.ones((size_tuple))

print("Applying noise to the array...")
np.random.seed(0)
Noise3DArray += np.random.normal(loc = 0.0, scale = 0.2, size = np.shape(Noise3DArray))

print("Applying Median Filter in 3D using the CPU...")
pars = {'input_data' : np.float32(Noise3DArray), # input grayscale image
        'kernel_size' : 5}

start_time = timeit.default_timer()
volume_filteredCPU = MEDIAN_FILT(pars['input_data'], pars['kernel_size'])
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

pars = {'input_data' : np.float32(Noise3DArray), # input grayscale image
        'radius' : 2}

print("Applying Median Filter in 3D using the GPU (global memory)...")
start_time = timeit.default_timer()
volume_filteredGPU = MEDIAN_FILT_GPU(pars['input_data'], pars['radius'])
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

print("Applying Median Filter in 3D using the GPU (shared memory)...")
start_time = timeit.default_timer()
volume_filteredGPU_shared = MEDIAN_FILT_GPU_SHARED(pars['input_data'], pars['radius'])
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

print("The sum of the CPU and GPU residual must be zero and  it equals to:")
print(np.sum(volume_filteredCPU - volume_filteredGPU))

midval = (int)(size_tuple[1]/2)
matplotlib.image.imsave('noisy_image.png', Noise3DArray[:,midval,:])
matplotlib.image.imsave('denoisedCPU.png', volume_filteredCPU[:,midval,:])
matplotlib.image.imsave('denoisedGPU.png', volume_filteredGPU[:,midval,:])
matplotlib.image.imsave('denoisedGPUshared.png', volume_filteredGPU_shared[:,midval,:])
matplotlib.image.imsave('denoised_CPU_GPU_residual.png', np.abs(volume_filteredCPU[:,midval,:] - volume_filteredGPU[:,midval,:]))
#IPython.embed()

