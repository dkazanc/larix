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

cdef extern int Autocrop_main(float *Input, float *mask_box, float *crop_indeces, float threshold, int margin_skip, int statbox_size, int increase_crop, int dimX, int dimY, int dimZ);
cdef extern int medianfilter_main_float(float *Input, float *Output, int radius, float mu_threshold, int ncores, int dimX, int dimY, int dimZ);
cdef extern int medianfilter_main_uint16(unsigned short *Input, unsigned short *Output, int radius, float mu_threshold, int ncores, int dimX, int dimY, int dimZ);
cdef extern int Diffusion_Inpaint_CPU_main(float *Input, unsigned char *Mask, float *Output, float lambdaPar, float sigmaPar, int iterationsNumb, float tau, int penaltytype, int dimX, int dimY, int dimZ);
cdef extern int NonlocalMarching_Inpaint_main(float *Input, unsigned char *M, float *Output, unsigned char *M_upd, int SW_increment, int iterationsNumb, int trigger, int dimX, int dimY, int dimZ);
cdef extern int Inpaint_simple_CPU_main(float *Input, unsigned char *Mask, float *Output, int iterations, int W_halfsize, int method_type, int ncores, int dimX, int dimY, int dimZ);
cdef extern int StripeWeights_main(float *input, float *output, int detectors_window_height, int detectors_window_width, int angles_window_depth, int vertical_mean_window, int ncores, long angl_size, long det_X_size, long det_Y_size);
cdef extern int StripesMergeMask_main(unsigned char *input, unsigned char *output, int stripe_width_max, int dilate, int ncores, long angl_size, long det_X_size, long det_Y_size);
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

    if (Autocrop_main(&Input[0,0], &mask[0,0], &crop_val_ar[0], threshold, margin_skip, statbox_size, increase_crop, dims[1], dims[0], 1)==0):
        return crop_val_ar
    else:
        ValueError("2D CPU autocrop function failed to return 0")

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

    if (Autocrop_main(&Input[0,0,0], &mask[0,0,0], &crop_val_ar[0], threshold, margin_skip, statbox_size, increase_crop, dims[2], dims[1], dims[0])==0):
        return crop_val_ar
    else:
        ValueError("3D CPU autocrop function failed to return 0")

#################################################################################
##############################Median Filtering ##################################
#################################################################################
def MEDIAN_FILT(Input, radius, ncores=0):
    input_type = Input.dtype
    if ((Input.ndim == 2) and (input_type == 'float32')):
        return MEDIAN_FILT_float32_2D(Input, radius, ncores)
    elif ((Input.ndim == 2) and (input_type == 'uint16')):
        return MEDIAN_FILT_uint16_2D(Input, radius, ncores)
    elif ((Input.ndim == 3) and (input_type == 'float32')):
        return  MEDIAN_FILT_float32_3D(Input, radius, ncores)
    elif ((Input.ndim == 3) and (input_type == 'uint16')):
        return  MEDIAN_FILT_uint16_3D(Input, radius, ncores)

def MEDIAN_FILT_float32_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] Input,
                            int radius,
                            int ncores):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] Output = \
            np.zeros([dims[0],dims[1]], dtype='float32')

    if ((radius  == 1) or (radius  == 2) or (radius  == 3) or (radius == 4) or (radius == 5)):
        if (medianfilter_main_float(&Input[0,0], &Output[0,0], radius, 0.0, ncores, dims[1], dims[0], 1)==0):
            return Output
        else:
            raise ValueError("2D CPU median filter function failed to return 0")
    else:
        print("Accepted kernel sizes are 1, 2, 3, 4, and 5")

def MEDIAN_FILT_uint16_2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] Input,
                            int radius,
                            int ncores):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] Output = \
            np.zeros([dims[0],dims[1]], dtype='uint16')

    if ((radius  == 1) or (radius  == 2) or (radius  == 3) or (radius == 4) or (radius == 5)):
        if (medianfilter_main_uint16(&Input[0,0], &Output[0,0], radius, 0.0, ncores, dims[1], dims[0], 1)==0):
            return Output
        else:
            raise ValueError("2D CPU median filter function failed to return 0")
    else:
        print("Accepted kernel sizes are 1, 2, 3, 4, and 5")

def MEDIAN_FILT_float32_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] Input,
                   int radius,
                   int ncores):

    cdef long dims[3]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]
    dims[2] = Input.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] Output = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')

    if ((radius  == 1) or (radius  == 2) or (radius  == 3) or (radius == 4) or (radius == 5)):
        if (medianfilter_main_float(&Input[0,0,0], &Output[0,0,0], radius, 0.0, ncores, dims[2], dims[1], dims[0])==0):
            return Output
        else:
            raise ValueError("3D CPU median filter function failed to return 0")
    else:
        print("Accepted kernel sizes are 1, 2, 3, 4, and 5")

def MEDIAN_FILT_uint16_3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] Input,
                          int radius,
                          int ncores):

    cdef long dims[3]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]
    dims[2] = Input.shape[2]

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] Output = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='uint16')

    if ((radius  == 1) or (radius  == 2) or (radius  == 3) or (radius == 4) or (radius == 5)):
        if (medianfilter_main_uint16(&Input[0,0,0], &Output[0,0,0], radius, 0.0, ncores, dims[2], dims[1], dims[0])==0):
            return Output
        else:
            raise ValueError("3D CPU median filter function failed to return 0")
    else:
        print("Accepted kernel sizes are 1, 2, 3, 4, and 5")

#################################################################################
##############################Median Dezingering ################################
#################################################################################
def MEDIAN_DEZING(Input, radius, mu_threshold, ncores=0):
    input_type = Input.dtype
    if ((Input.ndim == 2) and (input_type == 'float32')):
        return MEDIAN_DEZING_float32_2D(Input, radius, mu_threshold, ncores)
    elif ((Input.ndim == 2) and (input_type == 'uint16')):
        return MEDIAN_DEZING_uint16_2D(Input, radius, mu_threshold, ncores)
    elif ((Input.ndim == 3) and (input_type == 'float32')):
        return  MEDIAN_DEZING_float32_3D(Input, radius, mu_threshold, ncores)
    elif ((Input.ndim == 3) and (input_type == 'uint16')):
        return  MEDIAN_DEZING_uint16_3D(Input, radius, mu_threshold, ncores)

def MEDIAN_DEZING_float32_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] Input,
                            int radius,
                            float mu_threshold,
                            int ncores):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] Output = \
            np.zeros([dims[0],dims[1]], dtype='float32')

    if ((radius  == 1) or (radius  == 2) or (radius  == 3) or (radius == 4) or (radius == 5)):
        if (medianfilter_main_float(&Input[0,0], &Output[0,0], radius, mu_threshold, ncores, dims[1], dims[0], 1)==0):
            return Output
        else:
            raise ValueError("2D CPU dezinger filter function failed to return 0")
    else:
        print("Accepted kernel sizes are 1, 2, 3, 4, and 5")

def MEDIAN_DEZING_uint16_2D(np.ndarray[np.uint16_t, ndim=2, mode="c"] Input,
                            int radius,
                            float mu_threshold,
                            int ncores):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.uint16_t, ndim=2, mode="c"] Output = \
            np.zeros([dims[0],dims[1]], dtype='uint16')


    if ((radius  == 1) or (radius  == 2) or (radius  == 3) or (radius == 4) or (radius == 5)):
        if (medianfilter_main_uint16(&Input[0,0], &Output[0,0], radius, mu_threshold, ncores, dims[1], dims[0], 1)==0):
            return Output
        else:
            raise ValueError("2D CPU dezinger filter function failed to return 0")
    else:
        print("Accepted kernel sizes are 1, 2, 3, 4, and 5")

def MEDIAN_DEZING_float32_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] Input,
                            int radius,
                            float mu_threshold,
                            int ncores):
    cdef long dims[3]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]
    dims[2] = Input.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] Output = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')

    if ((radius  == 1) or (radius  == 2) or (radius  == 3) or (radius == 4) or (radius == 5)):
        if (medianfilter_main_float(&Input[0,0,0], &Output[0,0,0], radius, mu_threshold, ncores, dims[2], dims[1], dims[0])==0):
            return Output
        else:
            raise ValueError("3D CPU dezinger filter function failed to return 0")
    else:
        print("Accepted kernel sizes are 1, 2, 3, 4, and 5")

def MEDIAN_DEZING_uint16_3D(np.ndarray[np.uint16_t, ndim=3, mode="c"] Input,
                            int radius,
                            float mu_threshold,
                            int ncores):

    cdef long dims[3]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]
    dims[2] = Input.shape[2]

    cdef np.ndarray[np.uint16_t, ndim=3, mode="c"] Output = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='uint16')

    if ((radius  == 1) or (radius  == 2) or (radius  == 3) or (radius == 4) or (radius == 5)):
        if (medianfilter_main_uint16(&Input[0,0,0], &Output[0,0,0], radius, mu_threshold, ncores, dims[2], dims[1], dims[0])==0):
            return Output
        else:
            raise ValueError("3D CPU dezinger filter function failed to return 0")
    else:
        print("Accepted kernel sizes are 1, 2, 3, 4, and 5")

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
    if (Diffusion_Inpaint_CPU_main(&inputData[0,0], &maskData[0,0], &outputData[0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type, dims[1], dims[0], 1)==0):
        return outputData
    else:
        raise ValueError("2D CPU nonlinear diffusion inpainting failed to return 0")

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
    if (Diffusion_Inpaint_CPU_main(&inputData[0,0,0], &maskData[0,0,0], &outputData[0,0,0], regularisation_parameter, edge_parameter, iterationsNumb, time_marching_parameter, penalty_type, dims[2], dims[1], dims[0])==0):
        return outputData
    else:
        raise ValueError("3D CPU nonlinear diffusion inpainting failed to return 0")
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
    if (NonlocalMarching_Inpaint_main(&inputData[0,0], &maskData[0,0], &outputData[0,0], &maskData_upd[0,0], SW_increment, iterationsNumb, 1, dims[1], dims[0], 1)==0):
        return (outputData, maskData_upd)
    else:
        raise ValueError("2D CPU nonlocal marching inpainting failed to return 0")

#*********************Inpainting WITH****************************#
#*********************Euclidian Weighting************************#
#****************************************************************#
def INPAINT_EUCL_WEIGHTED(inputData, maskData, iterationsNumb, windowsize_half, method_type, ncores=0):
    if (method_type == 'median'):
        method_type_int = 1
    elif (method_type == 'random'):
        method_type_int = 2
    else: 
        method_type_int = 0
    if inputData.ndim == 2:
        return INPAINT_EUC_WEIGHT_2D(inputData, maskData, iterationsNumb, windowsize_half, method_type_int, ncores)
    elif inputData.ndim == 3:
        return INPAINT_EUC_WEIGHT_3D(inputData, maskData, iterationsNumb, windowsize_half, method_type_int, ncores)
def INPAINT_EUC_WEIGHT_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData,
               np.ndarray[np.uint8_t, ndim=2, mode="c"] maskData,
               int iterationsNumb,
               int windowsize_half,
               int method_type_int, 
               int ncores):

    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
            np.zeros([dims[0],dims[1]], dtype='float32')

    if (Inpaint_simple_CPU_main(&inputData[0,0], &maskData[0,0], &outputData[0,0], iterationsNumb, windowsize_half, method_type_int, ncores, dims[1], dims[0], 1)==0):
        return outputData
    else:
        raise ValueError("2D CPU inpainting failed to return 0")
def INPAINT_EUC_WEIGHT_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData,
               np.ndarray[np.uint8_t, ndim=3, mode="c"] maskData,
               int iterationsNumb,
               int windowsize_half,
               int method_type_int, 
               int ncores):

    cdef long dims[3]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]
    dims[2] = inputData.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] outputData = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')

    if (Inpaint_simple_CPU_main(&inputData[0,0,0], &maskData[0,0,0], &outputData[0,0,0], iterationsNumb, windowsize_half, method_type_int, ncores, dims[2], dims[1], dims[0])==0):
        return outputData
    else:
        raise ValueError("3D CPU inpainting failed to return 0")
    
#****************************************************************#
#*********************Stripes detection**************************#
#****************************************************************#
def STRIPES_DETECT(inputData, search_window_dims=(1,5,1), horiz_window_size = 5, ncores=0):
    """Method to detect stripes in sinograms (2D) OR projection data (3D). The method involves 3 steps:
    1. Taking first derrivative of the input in the direction orthogonal to stripes.
    2. Slide horizontal rectangular window orthogonal to stripes direction to accenuate outliers (stripes) using median.
    3. Slide the vertical thin (1 pixel) window to calculate a mean (further accenuates stripes).

    Args:
        inputData (array): sinogram (2D) [angles x detectorsX] OR projection data (3D) [detectorsX x angles x detectorsY]
        search_window_dims (tuple, optional): (detectors_window_height, detectors_window_width, angles_window_depth). Defaults to (1,5,1).        
        horiz_window_size (int, optional): the half size of the horizontal 1D window to calculate mean
        ncores (int, optional): the number of CPU cores. Defaults to 0.

    Returns:
        array: Calculated weights
    """    
    detectors_window_height = search_window_dims[0]
    detectors_window_width = search_window_dims[1]
    angles_window_depth = search_window_dims[2]
    if inputData.ndim == 2:
        return __STRIPES_DETECT_2D(inputData, detectors_window_height, detectors_window_width, 1, horiz_window_size, ncores)
    elif inputData.ndim == 3:
        return 0
def __STRIPES_DETECT_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData,
               int detectors_window_height,
               int detectors_window_width,
               int angles_window_depth,
               int horiz_window_size, 
               int ncores):

    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] outputData = \
            np.zeros([dims[0],dims[1]], dtype='float32')

    if (StripeWeights_main(&inputData[0,0], &outputData[0,0], detectors_window_height, detectors_window_width, angles_window_depth, horiz_window_size, ncores, dims[1], dims[0], 1)==0):
        return outputData
    else:
        raise ValueError("2D CPU stripe detection failed to return 0")

def STRIPES_MERGE(inputData, stripe_width_max_perc=3, dilate=1, ncores=0):
    """Method to merge two stripes in the distance defined by stripe_width_max

    Args:
        inputData (array): uint8 mask array where double stripes are present with ones
        stripe_width_max_perc (float, optional): the maximum width of stripes in the data, given in percents relative to the size of the DetectorX
        dilate (int, optional): the number of pixels/voxels to dilate the obtained mask
        ncores (int, optional): the number of CPU cores. Defaults to 0.

    Returns:
        array: uint8 processed mask
    """
    det_sizeX = np.size(inputData,1)
    stripe_width_max = (0.01*stripe_width_max_perc)*det_sizeX # the max strip width in pixels
    if inputData.ndim == 2:
        return __STRIPES_MERGE_2D(inputData, stripe_width_max, dilate, ncores)
    elif inputData.ndim == 3:
        return 0
def __STRIPES_MERGE_2D(np.ndarray[np.uint8_t, ndim=2, mode="c"] inputData,
               int stripe_width_max,
               int dilate,
               int ncores):

    cdef long dims[2]
    dims[0] = inputData.shape[0]
    dims[1] = inputData.shape[1]

    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] outputData = \
            np.zeros([dims[0],dims[1]], dtype='uint8')

    if (StripesMergeMask_main(&inputData[0,0], &outputData[0,0], stripe_width_max, dilate, ncores, dims[1], dims[0], 1)==0):
        return outputData
    else:
        raise ValueError("2D CPU stripe merge failed to return 0")       