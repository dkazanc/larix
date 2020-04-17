# distutils: language=c++
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

cdef extern int MedianFilt_GPU_main(float *Input, float *Output, int filter_half_window_size, float mu_threshold, int N, int M, int Z);
#################################################################################
###########################Median Filtering (GPU) ###############################
#################################################################################
def MEDIAN_FILT_GPU(Input, filter_half_window_size):
    if Input.ndim == 2:
        return MEDIAN_FILT_GPU_2D(Input, filter_half_window_size)
    elif Input.ndim == 3:
        return 0

def MEDIAN_FILT_GPU_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] Input,
                    int filter_half_window_size):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] Output = \
            np.zeros([dims[0],dims[1]], dtype='float32')

    if (MedianFilt_GPU_main(&Input[0,0], &Output[0,0], filter_half_window_size, 0.0, dims[1], dims[0], 1)==0):
        return Output
    else:
        raise ValueError(CUDAErrorMessage)

#################################################################################
#########################Median Dezingering (GPU) ###############################
#################################################################################
def MEDIAN_DEZING_GPU(Input, filter_half_window_size, mu_threshold):
    if Input.ndim == 2:
        return MEDIAN_DEZING_GPU_2D(Input, filter_half_window_size, mu_threshold)
    elif Input.ndim == 3:
        return 0

def MEDIAN_DEZING_GPU_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] Input,
                    int filter_half_window_size,
                    float mu_threshold):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] Output = \
            np.zeros([dims[0],dims[1]], dtype='float32')

    if (MedianFilt_GPU_main(&Input[0,0], &Output[0,0], filter_half_window_size, mu_threshold, dims[1], dims[0], 1) == 0):
        return Output
    else:
        raise ValueError(CUDAErrorMessage)
