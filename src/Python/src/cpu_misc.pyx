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
