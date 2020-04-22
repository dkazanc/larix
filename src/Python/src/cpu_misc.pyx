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

cdef extern float Autocrop_main(float *Input, float *mask_box, float *crop_indeces, float threshold, int margin_skip, int statbox_size, int increase_crop, int dimX, int dimY, int dimZ);
cdef extern float medianfilter_main(float *Input, float *Output, int kernel_size, float mu_threshold, int dimX, int dimY, int dimZ);
#################################################################
##########################Autocropper ###########################
#################################################################
def AUTOCROP(Input, threshold, margin_skip, statbox_size, increase_crop):
    if Input.ndim == 2:
        return AUTOCROP_2D(Input, threshold, margin_skip, statbox_size, increase_crop)
    elif Input.ndim == 3:
        return AUTOCROP_3D(Input, threshold, margin_skip, statbox_size, increase_crop)

def AUTOCROP_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] Input,
	float threshold,
	int margin_skip,
	int statbox_size,
	int increase_crop):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] mask = \
            np.zeros([dims[0],dims[1]], dtype='float32')

    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] crop_val_ar = \
            np.zeros([4], dtype='float32')

    Autocrop_main(&Input[0,0], &mask[0,0], &crop_val_ar[0], threshold, margin_skip, statbox_size, increase_crop, dims[1], dims[0], 1)
    return crop_val_ar

def AUTOCROP_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] Input,
	float threshold,
	int margin_skip,
	int statbox_size,
	int increase_crop):

    cdef long dims[3]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]
    dims[2] = Input.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] mask = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')

    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] crop_val_ar = \
            np.zeros([4], dtype='float32')

    Autocrop_main(&Input[0,0,0], &mask[0,0,0], &crop_val_ar[0], threshold, margin_skip, statbox_size, increase_crop, dims[2], dims[1], dims[0])
    return crop_val_ar

#################################################################################
##############################Median Filtering ##################################
#################################################################################
def MEDIAN_FILT(Input, kernel_size):
    if Input.ndim == 2:
        return MEDIAN_FILT_2D(Input, kernel_size)
    elif Input.ndim == 3:
        return MEDIAN_FILT_3D(Input, kernel_size)

def MEDIAN_FILT_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] Input,
                   int kernel_size):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] Output = \
            np.zeros([dims[0],dims[1]], dtype='float32')

    if ((kernel_size  == 3) or (kernel_size  == 5) or (kernel_size  == 7) or (kernel_size == 9) or (kernel_size == 11)):
        medianfilter_main(&Input[0,0], &Output[0,0], kernel_size, 0.0, dims[1], dims[0], 1)
    else:
        print("Accepted kernel sizes are 3, 5, 7, 9, and 11")
    return Output

def MEDIAN_FILT_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] Input,
                   int kernel_size):

    cdef long dims[3]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]
    dims[2] = Input.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] Output = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')

    if ((kernel_size  == 3) or (kernel_size  == 5) or (kernel_size  == 7) or (kernel_size == 9) or (kernel_size == 11)):
        medianfilter_main(&Input[0,0,0], &Output[0,0,0], kernel_size, 0.0, dims[2], dims[1], dims[0])
    else:
        print("Accepted kernel sizes are 3, 5, 7, 9, and 11")
    return Output
#################################################################################
##############################Median Dezingering ################################
#################################################################################
def MEDIAN_DEZING(Input, kernel_size, mu_threshold):
    if Input.ndim == 2:
        return MEDIAN_DEZING_2D(Input, kernel_size, mu_threshold)
    elif Input.ndim == 3:
        return  MEDIAN_DEZING_3D(Input, kernel_size, mu_threshold)

def MEDIAN_DEZING_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] Input,
                    int kernel_size,
                    float mu_threshold):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] Output = \
            np.zeros([dims[0],dims[1]], dtype='float32')

    if ((kernel_size  == 3) or (kernel_size  == 5) or (kernel_size  == 7) or (kernel_size == 9) or (kernel_size == 11)):
        medianfilter_main(&Input[0,0], &Output[0,0], kernel_size, mu_threshold, dims[1], dims[0], 1)
    else:
        print("Accepted kernel sizes are 3, 5, 7, 9, and 11")
    return Output

def MEDIAN_DEZING_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] Input,
                    int kernel_size,
                    float mu_threshold):

    cdef long dims[3]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]
    dims[2] = Input.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] Output = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')

    if ((kernel_size  == 3) or (kernel_size  == 5) or (kernel_size  == 7) or (kernel_size == 9) or (kernel_size == 11)):
        medianfilter_main(&Input[0,0,0], &Output[0,0,0], kernel_size, mu_threshold, dims[2], dims[1], dims[0])
    else:
        print("Accepted kernel sizes are 3, 5, 7, 9, and 11")
    return Output