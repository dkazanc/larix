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

cdef extern int MedianFilt_GPU_main_float32(float *Input, float *Output, int kernel_size, float mu_threshold, int N, int M, int Z);
cdef extern int MedianFilt_GPU_main_uint16(unsigned short *Input, unsigned short *Output, int kernel_size, float mu_threshold, int N, int M, int Z);
#################################################################################
###########################Median Filtering (GPU) ###############################
#################################################################################
def MEDIAN_FILT_GPU(Input, kernel_size):
    input_type = Input.dtype
    if ((Input.ndim == 2) and (input_type == 'float32')):
        return MEDIAN_FILT_GPU_float32_2D(Input, kernel_size)
    elif ((Input.ndim == 2) and (input_type == 'uint16')):
        return MEDIAN_FILT_GPU_uint16_2D(Input, kernel_size)
    elif ((Input.ndim == 3) and (input_type == 'float32')):
        return  MEDIAN_FILT_GPU_float32_3D(Input, kernel_size)
    elif ((Input.ndim == 3) and (input_type == 'uint16')):
        return  MEDIAN_FILT_GPU_uint16_3D(Input, kernel_size)

def MEDIAN_FILT_GPU_float32_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] Input,
                    int kernel_size):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] Output = \
            np.zeros([dims[0],dims[1]], dtype='float32')

    if ((kernel_size  == 3) or (kernel_size  == 5) or (kernel_size  == 7) or (kernel_size == 9) or (kernel_size == 11)):
        if (MedianFilt_GPU_main_float32(&Input[0,0], &Output[0,0], kernel_size, 0.0, dims[1], dims[0], 1)==0):
            return Output
        else:
            raise ValueError(CUDAErrorMessage)
    else:
        print("Accepted kernel sizes are 3, 5, 7, 9, and 11")

def MEDIAN_FILT_GPU_uint16_2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] Input,
                    int kernel_size):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] Output = \
            np.zeros([dims[0],dims[1]], dtype='uint16')

    if ((kernel_size  == 3) or (kernel_size  == 5) or (kernel_size  == 7) or (kernel_size == 9) or (kernel_size == 11)):
        if (MedianFilt_GPU_main_uint16(&Input[0,0], &Output[0,0], kernel_size, 0.0, dims[1], dims[0], 1)==0):
            return Output
        else:
            raise ValueError(CUDAErrorMessage)
    else:
        print("Accepted kernel sizes are 3, 5, 7, 9, and 11")

def MEDIAN_FILT_GPU_float32_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] Input,
                    int kernel_size):

    cdef long dims[3]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]
    dims[2] = Input.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] Output = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')

    if ((kernel_size  == 3) or (kernel_size  == 5) or (kernel_size  == 7) or (kernel_size == 9) or (kernel_size == 11)):
        if (MedianFilt_GPU_main_float32(&Input[0,0,0], &Output[0,0,0], kernel_size, 0.0, dims[2], dims[1], dims[0])==0):
            return Output
        else:
            raise ValueError(CUDAErrorMessage)
    else:
        print("Accepted kernel sizes are 3, 5, 7, 9, and 11")


def MEDIAN_FILT_GPU_uint16_3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] Input,
                              int kernel_size):

    cdef long dims[3]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]
    dims[2] = Input.shape[2]

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] Output = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='uint16')

    if ((kernel_size  == 3) or (kernel_size  == 5) or (kernel_size  == 7) or (kernel_size == 9) or (kernel_size == 11)):
        if (MedianFilt_GPU_main_uint16(&Input[0,0,0], &Output[0,0,0], kernel_size, 0.0, dims[2], dims[1], dims[0])==0):
            return Output
        else:
            raise ValueError(CUDAErrorMessage)
    else:
        print("Accepted kernel sizes are 3, 5, 7, 9, and 11")
#################################################################################
#########################Median Dezingering (GPU) ###############################
#################################################################################
def MEDIAN_DEZING_GPU(Input, kernel_size, mu_threshold):
    input_type = Input.dtype
    if ((Input.ndim == 2) and (input_type == 'float32')):
        return MEDIAN_DEZING_float32_GPU_2D(Input, kernel_size, mu_threshold)
    elif ((Input.ndim == 2) and (input_type == 'uint16')):
        return MEDIAN_DEZING_uint16_GPU_2D(Input, kernel_size, mu_threshold)
    elif ((Input.ndim == 3) and (input_type == 'float32')):
        return  MEDIAN_DEZING_float32_GPU_3D(Input, kernel_size, mu_threshold)
    elif ((Input.ndim == 3) and (input_type == 'uint16')):
        return  MEDIAN_DEZING_uint16_GPU_3D(Input, kernel_size, mu_threshold)

def MEDIAN_DEZING_float32_GPU_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] Input,
                    int kernel_size,
                    float mu_threshold):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] Output = \
            np.zeros([dims[0],dims[1]], dtype='float32')

    if ((kernel_size  == 3) or (kernel_size  == 5) or (kernel_size  == 7) or (kernel_size == 9) or (kernel_size == 11)):
        if (MedianFilt_GPU_main_float32(&Input[0,0], &Output[0,0], kernel_size, mu_threshold, dims[1], dims[0], 1)==0):
            return Output
        else:
            raise ValueError(CUDAErrorMessage)
    else:
        print("Accepted kernel sizes are 3, 5, 7, 9, and 11")

def MEDIAN_DEZING_uint16_GPU_2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] Input,
                    int kernel_size,
                    float mu_threshold):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] Output = \
            np.zeros([dims[0],dims[1]], dtype='uint16')

    if ((kernel_size  == 3) or (kernel_size  == 5) or (kernel_size  == 7) or (kernel_size == 9) or (kernel_size == 11)):
        if (MedianFilt_GPU_main_uint16(&Input[0,0], &Output[0,0], kernel_size, mu_threshold, dims[1], dims[0], 1)==0):
            return Output
        else:
            raise ValueError(CUDAErrorMessage)
    else:
        print("Accepted kernel sizes are 3, 5, 7, 9, and 11")


def MEDIAN_DEZING_float32_GPU_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] Input,
                    int kernel_size,
                    float mu_threshold):

    cdef long dims[3]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]
    dims[2] = Input.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] Output = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')

    if ((kernel_size  == 3) or (kernel_size  == 5) or (kernel_size  == 7) or (kernel_size == 9) or (kernel_size == 11)):
        if (MedianFilt_GPU_main_float32(&Input[0,0,0], &Output[0,0,0], kernel_size, mu_threshold, dims[2], dims[1], dims[0])==0):
            return Output
        else:
            raise ValueError(CUDAErrorMessage)
    else:
        print("Accepted kernel sizes are 3, 5, 7, 9, and 11")


def MEDIAN_DEZING_uint16_GPU_3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] Input,
                    int kernel_size,
                    float mu_threshold):

    cdef long dims[3]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]
    dims[2] = Input.shape[2]

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] Output = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='uint16')

    if ((kernel_size  == 3) or (kernel_size  == 5) or (kernel_size  == 7) or (kernel_size == 9) or (kernel_size == 11)):
        if (MedianFilt_GPU_main_uint16(&Input[0,0,0], &Output[0,0,0], kernel_size, mu_threshold, dims[2], dims[1], dims[0])==0):
            return Output
        else:
            raise ValueError(CUDAErrorMessage)
    else:
        print("Accepted kernel sizes are 3, 5, 7, 9, and 11")
