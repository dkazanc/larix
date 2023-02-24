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
cdef extern int stripesdetect3d_main_float(float* Input, float* Output, int window_halflength_vertical, int ratio_radius, int ncores, long dimX, long dimY, long dimZ);
cdef extern int stripesmask3d_main_float(float* Input, unsigned char* Output, float threshold_val, int stripe_length_min, int stripe_depth_min, int stripe_width_min, float sensitivity, int ncores, long dimX, long dimY, long dimZ);
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
        return __INPAINT_EUC_WEIGHT_2D(np.ascontiguousarray(inputData, dtype=np.float32), np.ascontiguousarray(maskData, dtype=np.uint8), iterationsNumb, windowsize_half, method_type_int, ncores)
    elif inputData.ndim == 3:
        return __INPAINT_EUC_WEIGHT_3D(np.ascontiguousarray(inputData, dtype=np.float32), np.ascontiguousarray(maskData, dtype=np.uint8), iterationsNumb, windowsize_half, method_type_int, ncores)
def __INPAINT_EUC_WEIGHT_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] inputData,
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
def __INPAINT_EUC_WEIGHT_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData,
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
def STRIPES_DETECT(inputData, size=10, radius=3, ncore=0):
    """
    Apply a stripes detection method to empasize their edges in a 3D array.
    The input must be normalized projection data in range [0,1] and given in
    the following axis orientation [angles, detY(depth), detX (horizontal)]. With 
    this orientation, the stripes are the vertical features. The method works with
    full and partial stripes of constant ot varying intensity. 

    Parameters
    ----------
    inputData : ndarray
        3D tomographic data of float32 data type, normalized [0,1] and given in
        [angles, detY(depth), detX (horizontal)] axis orientation.
    size : int, optional
        The pixel size of the vertical 1D median filter to minimise false detections. Increase it if you have longer or full stripes in the data. 
    radius : int, optional
        The pixel size of the stencil to calculate the mean ratio between vertical and horizontal orientations. The larger values will enlarge the mask width.
    ncore : int, optional
        Number of cores that will be assigned to jobs. All cores will be used
        if unspecified.

    Returns
    -------
    ndarray
        Weights for stripe's edges as a 3D array of float32 data type. 
        The weights can be thresholded or passed to stripes_mask3d function to obtain a binary mask.

    Raises
    ------
    ValueError
        If the input array is not three dimensional.        
    """ 
 
    if inputData.ndim == 3:
        dz, dy, dx = inputData.shape
        if (dz == 0) or (dy == 0) or (dx == 0):
            raise ValueError("The length of one of dimensions is equal to zero")
    else:
        raise ValueError("The input array must be a 3D array")

    if size <= 0 or size > dz //  2:
        raise ValueError("The size of the filter should be larger than zero and smaller than the half of the vertical dimension")
    
    if inputData.ndim == 2:
        return 0
    elif inputData.ndim == 3:
        return __STRIPES_DETECT_3D(np.ascontiguousarray(inputData),
                                   size,
                                   radius,
                                   ncore,
                                   dx, dy, dz)

def __STRIPES_DETECT_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData,
                       int size,
                       int radius,
                       int ncore,
                       long dx,
                       long dy,
                       long dz):

    cdef long dims[3]
    dims[0] = dz
    dims[1] = dy
    dims[2] = dx

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] outputData = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')

    if (stripesdetect3d_main_float(&inputData[0,0,0], &outputData[0,0,0], size, radius, ncore, dims[2], dims[1], dims[0])==0):
        return (outputData)
    else:
        raise ValueError("3D CPU stripe detection failed to return 0")

#****************************************************************#
#****************Stripes mask generation*************************#
#****************************************************************#

def STRIPES_MERGE(weights, 
                  threshold = 0.6,
                  min_stripe_length = 20,
                  min_stripe_depth  = 10,
                  min_stripe_width = 5,
                  sensitivity_perc = 85.0,
                  ncore=0):
    """
    Takes the result of the stripes_detect3d module as an input and generates a 
    binary 3D mask with ones where stripes present. The method tries to eliminate
    non-stripe features in data by checking the weight consistency in three directions.

    Parameters
    ----------
    weights : ndarray
        3D weights array, a result of stripes_detect3d module given in
        [angles, detY(depth), detX] axis orientation.
    threshold : float, optional
        Threshold for the given weights, the smaller values correspond to the stripes
    min_stripe_length : int, optional
        Minimum accepted length of a stripe in pixels. Can be large if there are full stripes in the data.
    min_stripe_depth : int, optional
        Minimum accepted depth of a stripe in pixels. The stripes do not extend very deep, with this parameter more non-stripe features can be removed. 
    min_stripe_width : int, optional
        Minimum accepted width of a stripe in pixels. The stripes can be merged together with this parameter.
    sensitivity_perc : float, optional
        The value in percents to impose less strict conditions on length, depth and width of a stripe.
    ncore : int, optional
        Number of cores that will be assigned to jobs. All cores will be used
        if unspecified.

    Returns
    -------
    ndarray
        A binary mask of uint8 data type with stripes highlighted.

    Raises
    ------
    ValueError
        If the input array is not three dimensional.

    """
 
    if weights.ndim == 3:
        dz, dy, dx = weights.shape
        if (dz == 0) or (dy == 0) or (dx == 0):
            raise ValueError("The length of one of dimensions is equal to zero")
    else:
        raise ValueError("The input array must be a 3D array")

    if min_stripe_length <= 0 or min_stripe_length >= dz:
        raise ValueError("The minimum length of a stripe cannot be zero or exceed the size of the angular dimension")

    if min_stripe_depth <= 0 or min_stripe_depth >= dy:
        raise ValueError("The minimum depth of a stripe cannot be zero or exceed the size of the depth dimension")

    if min_stripe_width <= 0 or min_stripe_width >= dx:
        raise ValueError("The minimum width of a stripe cannot be zero or exceed the size of the horizontal dimension")

    if 0.0 < sensitivity_perc <= 100.0:
        pass
    else:
        raise ValueError("sensitivity_perc value must be in (0, 100] percentage range ")
    
    if weights.ndim == 2:
        return 0
    elif weights.ndim == 3:
        return __STRIPES_MERGE_3D(np.ascontiguousarray(weights), 
                                  threshold,
                                  min_stripe_length,
                                  min_stripe_depth,
                                  min_stripe_width,
                                  sensitivity_perc,
                                  ncore,
                                  dx, dy, dz)

def __STRIPES_MERGE_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] inputData,
                       float threshold,
                       int min_stripe_length,
                       int min_stripe_depth,
                       int min_stripe_width,
                       float sensitivity_perc,
                       int ncore,
                       long dx,
                       long dy,
                       long dz):

    cdef long dims[3]
    dims[0] = dz
    dims[1] = dy
    dims[2] = dx

    cdef np.ndarray[np.uint8_t, ndim=3, mode="c"] outputData = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='uint8')

    if (stripesmask3d_main_float(&inputData[0,0,0], 
                                 &outputData[0,0,0],
                                 threshold,
                                 min_stripe_length,
                                 min_stripe_depth,
                                 min_stripe_width,
                                 sensitivity_perc,
                                 ncore, dims[2], dims[1], dims[0])==0):
        return (outputData)
    else:
        raise ValueError("3D CPU stripe mask generation failed to return 0")