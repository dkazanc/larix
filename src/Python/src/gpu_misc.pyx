# distutils: language=c++
# cython: language_level=3
"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cython
import numpy as np
cimport numpy as np

CUDAErrorMessage = 'CUDA error'

cdef extern int MedianFilt_global_GPU_main_float32(float *Input, float *Output, int radius, float mu_threshold, int gpu_device, int N, int M, int Z);
#cdef extern int MedianFilt_GPU_main_uint16(unsigned short *Input, unsigned short *Output, int kernel_size, float mu_threshold, int N, int M, int Z);
cdef extern int MedianFilt_shared_GPU_main_float32(float *Input, float *Output, int radius, int gpu_device, int N, int M, int Z);
#################################################################################
###########################Median Filtering (GPU) ###############################
#################################################################################
def MEDIAN_FILT_GPU_SHARED(Input, radius, *gpu_device_list):
    input_type = Input.dtype
    if not gpu_device_list:
        gpu_device = 0 # set to be a default 0th device
    else:
        gpu_device = gpu_device_list[0] # set to be a chosen GPU device
    if ((Input.ndim == 2) and (input_type == 'float32')):
        return MEDIAN_FILT_GPU_shared_float32_2D(Input, radius, gpu_device)
    elif ((Input.ndim == 2) and (input_type == 'uint16')):
        return 0
    elif ((Input.ndim == 3) and (input_type == 'float32')):
        return MEDIAN_FILT_GPU_shared_float32_3D(Input, radius, gpu_device)
    elif ((Input.ndim == 3) and (input_type == 'uint16')):
        return  0

def MEDIAN_FILT_GPU_shared_float32_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] Input,
                    int radius, int gpu_device):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] Output = \
            np.zeros([dims[0],dims[1]], dtype='float32')

    if ((radius  == 1) or (radius  == 2) or (radius  == 3) or (radius == 4) or (radius == 5)):
        if (MedianFilt_shared_GPU_main_float32(&Input[0,0], &Output[0,0], radius, gpu_device, dims[1], dims[0], 1)==0):
            return Output
        else:
            raise ValueError(CUDAErrorMessage)
    else:
        print("Accepted kernel sizes are 1, 2, 3, 4, and 5")            
    
def MEDIAN_FILT_GPU_shared_float32_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] Input,
                    int radius, int gpu_device):

    cdef long dims[3]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]
    dims[2] = Input.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] Output = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')

    if ((radius  == 1) or (radius  == 2) or (radius  == 3) or (radius == 4) or (radius == 5)):
        if (MedianFilt_shared_GPU_main_float32(&Input[0,0,0], &Output[0,0,0], radius, gpu_device, dims[2], dims[1], dims[0])==0):
            return Output
        else:
            raise ValueError(CUDAErrorMessage)
    else:
        print("Accepted kernel sizes are 1, 2, 3, 4, and 5")    

########################## GLOBAL MEDIAN ##################################
def MEDIAN_FILT_GPU(Input, radius, *gpu_device_list):
    input_type = Input.dtype
    if not gpu_device_list:
        gpu_device = 0 # set to be a default 0th device
    else:
        gpu_device = gpu_device_list[0] # set to be a chosen GPU device
    if ((Input.ndim == 2) and (input_type == 'float32')):
        return MEDIAN_FILT_GPU_global_float32_2D(Input, radius, gpu_device)
    elif ((Input.ndim == 2) and (input_type == 'uint16')):
        return 0
    elif ((Input.ndim == 3) and (input_type == 'float32')):
        return MEDIAN_FILT_GPU_global_float32_3D(Input, radius, gpu_device)
    elif ((Input.ndim == 3) and (input_type == 'uint16')):
        return  0

def MEDIAN_FILT_GPU_global_float32_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] Input,
                    int radius, int gpu_device):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] Output = \
            np.zeros([dims[0],dims[1]], dtype='float32')

    if (MedianFilt_global_GPU_main_float32(&Input[0,0], &Output[0,0], radius, 0.0, gpu_device, dims[1], dims[0], 1)==0):
        return Output
    else:
        raise ValueError(CUDAErrorMessage)
    
def MEDIAN_FILT_GPU_global_float32_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] Input,
                    int radius, int gpu_device):

    cdef long dims[3]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]
    dims[2] = Input.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] Output = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')

    if (MedianFilt_global_GPU_main_float32(&Input[0,0,0], &Output[0,0,0], radius, 0.0, gpu_device, dims[2], dims[1], dims[0])==0):
        return Output
    else:
        raise ValueError(CUDAErrorMessage)    