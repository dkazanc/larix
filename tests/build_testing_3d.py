import numpy as np
import timeit
import bz2
#import IPython

from larix.methods.misc import MEDIAN_FILT
from larix.methods.misc_gpu import MEDIAN_FILT_GPU

# load the volume data
print("Decompressing data3D_to_crop.npy.bz2 file containing noisy volume...")
with bz2.BZ2File('data3D_to_crop.npy.bz2', 'r') as f:
    volume = np.load(f, allow_pickle=True)

print("Decompressing volume_filteredCPU.npy.bz2 file containing CPU benchmark...")
with bz2.BZ2File('volume_filteredCPU.npy.bz2', 'r') as f:
    volume_filteredCPU = np.load(f, allow_pickle=True)

pars = {'input_data' : volume, # input a grayscale image
        'kernel_size' : 3}


#start_time = timeit.default_timer()
#volume_filteredCPU = MEDIAN_FILT(pars['input_data'], pars['kernel_size'])
#txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
#print (txtstr)


print("Applying Median Filter in 3D using GPU...")
start_time = timeit.default_timer()
volume_filteredGPU = MEDIAN_FILT_GPU(pars['input_data'], pars['kernel_size'])
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

#IPython.embed()
# generate a result for 3D filtering on the CPU to have something to compare to
# when testing the GPU filtering
#volume_filteredCPU = MEDIAN_FILT(pars['input_data'], pars['kernel_size'])
# save filltered CPU volume
#np.save('volume_FilteredCPU.npy', volume_filteredCPU)

# define a specific horizontal slice of the filtered volume to compare the CPU
# and GPU results
slice_idx = 0

# calculate residual between 3D filtering with CPU vs GPU
single_slice_residual = np.sum(volume_filteredGPU[:, slice_idx, :] - volume_filteredCPU[:, slice_idx, :])
print(f"The sum of the CPU and GPU residual for the horizontal slice {slice_idx} of the volume is {single_slice_residual}")

# printing other things to check
#print('Slice of CPU is:')
#print(volume_filteredCPU[:, slice_idx, :])
#print('Slice of GPU is:')
#print(volume_filteredGPU[:, slice_idx, :])
total_residual = np.sum(volume_filteredGPU - volume_filteredCPU)
print(f"The sum of the CPU and GPU total residual for the filtered volume is {total_residual}")

# save GPU filtered volume
np.save('volume_FilteredGPU.npy', volume_filteredGPU)
