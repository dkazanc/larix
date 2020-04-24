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
cdef extern float Diffusion_Inpaint_CPU_main(float *Input, unsigned char *Mask, float *Output, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, int penaltytype, int dimX, int dimY, int dimZ);
cdef extern float NonlocalMarching_Inpaint_main(float *Input, unsigned char *M, float *Output, unsigned char *M_upd, int SW_increment, int iterationsNumb, int trigger, int dimX, int dimY, int dimZ);
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

#*********************Inpainting WITH****************************#
#***************Nonlinear (Isotropic) Diffusion******************#
#****************************************************************#
def INPAINT_NDF(inputData, maskData, regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type):
    if inputData.ndim == 2:
        return INPAINT_NDF_2D(inputData, maskData, regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type)
    elif inputData.ndim == 3:
        return INPAINT_NDF_3D(inputData, maskData, regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type)

def INPAINT_NDF_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData,
                     np.ndarray[np.uint8_t, ndim=2, mode="c"] maskData,
                     float regularisation_parameter,
                     float edge_parameter,
                     int iterationsNumb,
                     float time_marching_parameter,
                     int penalty_type):

    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]


    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
            np.zeros([dims[0],dims[1]], dtype='float32')

    # Run Inpaiting by Diffusion iterations for 2D data
    Diffusion_Inpaint_CPU_main(&inputData[0,0], &maskData[0,0], &outputData[0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type, dims[1], dims[0], 1)
    return outputData

def INPAINT_NDF_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData,
                     np.ndarray[np.uint8_t, ndim=3, mode="c"] maskData,
                     float regularisation_parameter,
                     float edge_parameter,
                     int iterationsNumb,
                     float time_marching_parameter,
                     int penalty_type):
    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] outputData = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')

    # Run Inpaiting by Diffusion iterations for 3D data
    Diffusion_Inpaint_CPU_main(&inputData[0,0,0], &maskData[0,0,0], &outputData[0,0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type, dims[2], dims[1], dims[0])
    return outputData

#*********************Inpainting WITH****************************#
#******************Nonlocal  Marching method*********************#
#****************************************************************#
def INPAINT_NM(inputData, maskData, SW_increment, iterationsNumb):
    if inputData.ndim == 2:
        return INPAINT_NM_2D(inputData, maskData, SW_increment, iterationsNumb)
    elif inputData.ndim == 3:
        return

def INPAINT_NM_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData,
               np.ndarray[np.uint8_t, ndim=2, mode="c"] maskData,
                     int SW_increment,
                     int iterationsNumb):
    
    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
            np.zeros([dims[0],dims[1]], dtype='float32')

    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] maskData_upd = \
            np.zeros([dims[0],dims[1]], dtype='uint8')

    # Run Inpaiting by Nonlocal vertical marching method for 2D data
    NonlocalMarching_Inpaint_main(&inputData[0,0], &maskData[0,0], &outputData[0,0],
                                  &maskData_upd[0,0],
                                  SW_increment, iterationsNumb, 1, dims[1], dims[0], 1)
    return (outputData, maskData_upd)
