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

cdef extern float Mask_merge_main(unsigned char *MASK, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *SelClassesList, unsigned char *ComboClasses, int tot_combinations, int SelClassesList_length, int classesNumb, int CorrectionWindow, int iterationsNumb, int dimX, int dimY, int dimZ);
cdef extern float mask_morph_main(unsigned char *MASK, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, int primeClass, int CorrectionWindow, int iterationsNumb, int dimX, int dimY, int dimZ);
cdef extern float MASK_evolve_main(float *Input, unsigned char *MASK_in, unsigned char *MASK_out, float threhsold, int iterations, int connectivity, float value1, float value2, int dimX, int dimY, int dimZ);
cdef extern float MASK_evolve_conditional_main(float *Input, unsigned char *MASK_in, unsigned char *MASK_conditional, unsigned char *MASK_out, float threhsold, int iterations, int connectivity, float value1, float value2, int dimX, int dimY, int dimZ);
cdef extern float autocropper_main(float *Input, float *mask_box, float *crop_indeces, int margin_size, int statbox_size, int dimX, int dimY, int dimZ);
##############################################################################
#****************************************************************#
#********Mask (segmented image) correction module **************#
#****************************************************************#
def MASK_CORR(maskData, class_names, total_classesNum, restricted_combinations, CorrectionWindow, iterationsNumb):
    #select_classes_ar = np.uint8(np.array([3, 0, 1])) # convert a tuple to array
    # get main classes to work with
    select_classes_ar = np.array([])
    for obj in class_names:
        if (str(obj) is 'air'):
            select_classes_ar = np.append(select_classes_ar,0)
        if (str(obj) is 'loop'):
            select_classes_ar = np.append(select_classes_ar,1)
        if (str(obj) is 'crystal'):
            select_classes_ar = np.append(select_classes_ar,2)
        if (str(obj) is 'liquor'):
            select_classes_ar = np.append(select_classes_ar,3)
        if (str(obj) is 'artifacts'):
            select_classes_ar = np.append(select_classes_ar,4)
    select_classes_ar = np.uint8(select_classes_ar)

    # get restricted combinations of 3 items in each combination
    combo_classes_ar = np.array([])
    for obj in restricted_combinations:
        for name in obj:
            if (str(name) is 'air'):
                combo_classes_ar = np.append(combo_classes_ar,0)
            if (str(name) is 'loop'):
                combo_classes_ar = np.append(combo_classes_ar,1)
            if (str(name) is 'crystal'):
                combo_classes_ar = np.append(combo_classes_ar,2)
            if (str(name) is 'liquor'):
                combo_classes_ar = np.append(combo_classes_ar,3)
            if (str(name) is 'artifacts'):
                combo_classes_ar = np.append(combo_classes_ar,4)
    combo_classes_ar = np.uint8(combo_classes_ar)
    #print(combinations_classes_ar)
    if maskData.ndim == 2:
        return MASK_CORR_2D(maskData, select_classes_ar, combo_classes_ar, total_classesNum, CorrectionWindow, iterationsNumb)
    elif maskData.ndim == 3:
        return MASK_CORR_3D(maskData, select_classes_ar, combo_classes_ar, total_classesNum, CorrectionWindow, iterationsNumb)

def MASK_CORR_2D(np.ndarray[np.uint8_t, ndim=2, mode="c"] maskData,
                    np.ndarray[np.uint8_t, ndim=1, mode="c"] select_classes_ar,
                    np.ndarray[np.uint8_t, ndim=1, mode="c"] combo_classes_ar,
                     int total_classesNum,
                     int CorrectionWindow,
                     int iterationsNumb):

    cdef long dims[2]
    dims[0] = maskData.shape[0]
    dims[1] = maskData.shape[1]

    select_classes_length = select_classes_ar.shape[0]
    tot_combinations = (int)(combo_classes_ar.shape[0]/int(4))

    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] mask_upd = \
            np.zeros([dims[0],dims[1]], dtype='uint8')
    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] corr_regions = \
            np.zeros([dims[0],dims[1]], dtype='uint8')

    # Run the function to process given MASK
    Mask_merge_main(&maskData[0,0], &mask_upd[0,0],
                    &corr_regions[0,0], &select_classes_ar[0], &combo_classes_ar[0], tot_combinations, select_classes_length,
                    total_classesNum, CorrectionWindow,
                    iterationsNumb, dims[1], dims[0], 1)
    return mask_upd

def MASK_CORR_3D(np.ndarray[np.uint8_t, ndim=3, mode="c"] maskData,
                    np.ndarray[np.uint8_t, ndim=1, mode="c"] select_classes_ar,
                    np.ndarray[np.uint8_t, ndim=1, mode="c"] combo_classes_ar,
                     int total_classesNum,
                     int CorrectionWindow,
                     int iterationsNumb):

    cdef long dims[3]
    dims[0] = maskData.shape[0]
    dims[1] = maskData.shape[1]
    dims[2] = maskData.shape[2]

    select_classes_length = select_classes_ar.shape[0]
    tot_combinations = (int)(combo_classes_ar.shape[0]/int(4))

    cdef np.ndarray[np.uint8_t, ndim=3, mode="c"] mask_upd = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='uint8')
    cdef np.ndarray[np.uint8_t, ndim=3, mode="c"] corr_regions = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='uint8')

   # Run the function to process given MASK
    Mask_merge_main(&maskData[0,0,0], &mask_upd[0,0,0],
                    &corr_regions[0,0,0], &select_classes_ar[0], &combo_classes_ar[0], tot_combinations, select_classes_length,
                    total_classesNum, CorrectionWindow,
                    iterationsNumb, dims[2], dims[1], dims[0])
    return mask_upd

#################################################################
################Morphological mask processing####################
#################################################################

def MASK_MORPH(maskData, primeClass, correctionWindow, iterationsNumb):
    if maskData.ndim == 2:
        return MASK_MORPH_2D(maskData, primeClass, correctionWindow, iterationsNumb)
    elif maskData.ndim == 3:
        return MASK_MORPH_3D(maskData, primeClass, correctionWindow, iterationsNumb)

def MASK_MORPH_2D(np.ndarray[np.uint8_t, ndim=2, mode="c"] maskData,
	 	     int primeClass,
                     int correctionWindow,
                     int iterationsNumb):

    cdef long dims[2]
    dims[0] = maskData.shape[0]
    dims[1] = maskData.shape[1]

    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] mask_upd = \
            np.zeros([dims[0],dims[1]], dtype='uint8')
    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] corr_regions = \
            np.zeros([dims[0],dims[1]], dtype='uint8')

    # Run the function to process given MASK
    mask_morph_main(&maskData[0,0], &mask_upd[0,0],
                    &corr_regions[0,0], primeClass, correctionWindow,
                    iterationsNumb, dims[1], dims[0], 1)
    return mask_upd

def MASK_MORPH_3D(np.ndarray[np.uint8_t, ndim=3, mode="c"] maskData,
	 	     int primeClass,
                     int correctionWindow,
                     int iterationsNumb):

    cdef long dims[3]
    dims[0] = maskData.shape[0]
    dims[1] = maskData.shape[1]
    dims[2] = maskData.shape[2]

    cdef np.ndarray[np.uint8_t, ndim=3, mode="c"] mask_upd = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='uint8')
    cdef np.ndarray[np.uint8_t, ndim=3, mode="c"] corr_regions = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='uint8')

   # Run the function to process given MASK
    mask_morph_main(&maskData[0,0,0], &mask_upd[0,0,0],
                    &corr_regions[0,0,0], primeClass, correctionWindow,
                    iterationsNumb, dims[2], dims[1], dims[0])
    return mask_upd

#################################################################
##########################EVOLVING MASK##########################
#################################################################
def MASK_EVOLVE(Input, maskData, threhsold, iterationsNumb, connectivity, method):
    value1 = 0.0
    value2 = 1.0
    mult = np.multiply(Input,maskData)
    arr = mult[np.nonzero(mult)]
    if (str(method) is 'mean'):
        value1 = np.mean(arr)
        arr = np.abs(arr - value1)
        value2 = 1.4826*np.mean(arr)
    elif (str(method) is 'median'):
        value1 = np.median(arr)
        arr = np.abs(arr - value1)
        value2 = 1.4826*np.median(arr)
    elif (str(method) is 'value'):
        value1 = 0.0
        value2 = 1.0
    else:
        print("Please select the mask statistics collection method: 'mean', 'median' or 'value'")
    if maskData.ndim == 2:
        return MASK_EVOLVE_2D(Input, maskData, threhsold, iterationsNumb, connectivity, value1, value2)
    elif maskData.ndim == 3:
        return MASK_EVOLVE_3D(Input, maskData, threhsold, iterationsNumb, connectivity, value1, value2)

def MASK_EVOLVE_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] Input,
                    np.ndarray[np.uint8_t, ndim=2, mode="c"] MASK_in,
                     float threhsold,
                     int iterationsNumb,
                     int connectivity,
                     float value1,
                     float value2):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] MASK_out = \
            np.zeros([dims[0],dims[1]], dtype='uint8')

    MASK_evolve_main(&Input[0,0], &MASK_in[0,0], &MASK_out[0,0], threhsold,
                    iterationsNumb, connectivity, value1, value2, dims[1], dims[0], 1)
    return MASK_out

def MASK_EVOLVE_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] Input,
                    np.ndarray[np.uint8_t, ndim=3, mode="c"] MASK_in,
                     float threhsold,
                     int iterationsNumb,
                     int connectivity,
                     float value1,
                     float value2):

    cdef long dims[3]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]
    dims[2] = Input.shape[2]

    cdef np.ndarray[np.uint8_t, ndim=3, mode="c"] MASK_out = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='uint8')

    MASK_evolve_main(&Input[0,0,0], &MASK_in[0,0,0], &MASK_out[0,0,0], threhsold,
                    iterationsNumb, connectivity, value1, value2, dims[2], dims[1], dims[0])
    return MASK_out

#################################################################################
def MASK_CONDITIONAL_EVOLVE(Input, maskData, condmask, threhsold, iterationsNumb, connectivity, method):
    value1 = 0.0
    value2 = 1.0
    mult = np.multiply(Input,maskData)
    arr = mult[np.nonzero(mult)]
    if (str(method) is 'mean'):
        value1 = np.mean(arr)
        arr = np.abs(arr - value1)
        value2 = 1.4826*np.mean(arr)
    elif (str(method) is 'median'):
        value1 = np.median(arr)
        arr = np.abs(arr - value1)
        value2 = 1.4826*np.median(arr)
    elif (str(method) is 'value'):
        value1 = 0.0
        value2 = 1.0
    else:
        print("Please select the mask statistics collection method: 'mean', 'median' or 'value'")
    if maskData.ndim == 2:
        return MASK_CONDITIONAL_EVOLVE_2D(Input, maskData, condmask, threhsold, iterationsNumb, connectivity, value1, value2)
    elif maskData.ndim == 3:
        return MASK_CONDITIONAL_EVOLVE_3D(Input, maskData, condmask, threhsold, iterationsNumb, connectivity, value1, value2)

def MASK_CONDITIONAL_EVOLVE_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] Input,
                    np.ndarray[np.uint8_t, ndim=2, mode="c"] MASK_in,
                    np.ndarray[np.uint8_t, ndim=2, mode="c"] MASK_conditional,
                    float threhsold,
                    int iterationsNumb,
                    int connectivity,
                    float value1,
                    float value2):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] MASK_out = \
            np.zeros([dims[0],dims[1]], dtype='uint8')

    MASK_evolve_conditional_main(&Input[0,0], &MASK_in[0,0], &MASK_conditional[0,0], &MASK_out[0,0], threhsold,
                    iterationsNumb, connectivity, value1, value2, dims[1], dims[0], 1)
    return MASK_out

def MASK_CONDITIONAL_EVOLVE_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] Input,
                    np.ndarray[np.uint8_t, ndim=3, mode="c"] MASK_in,
                    np.ndarray[np.uint8_t, ndim=3, mode="c"] MASK_conditional,
                    float threhsold,
                    int iterationsNumb,
                    int connectivity,
                    float value1,
                    float value2):

    cdef long dims[3]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]
    dims[2] = Input.shape[2]

    cdef np.ndarray[np.uint8_t, ndim=3, mode="c"] MASK_out = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='uint8')

    MASK_evolve_conditional_main(&Input[0,0,0], &MASK_in[0,0,0], &MASK_conditional[0,0,0], &MASK_out[0,0,0], threhsold,
                    iterationsNumb, connectivity, value1, value2, dims[2], dims[1], dims[0])
    return MASK_out
#################################################################
##########################Autocropper ###########################
#################################################################
def AUTOCROP(Input, margin_size, statbox_size):
    if Input.ndim == 2:
        return AUTOCROP_2D(Input, margin_size, statbox_size)
    elif Input.ndim == 3:
        return 0

def AUTOCROP_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] Input,
                np.ndarray[np.float32_t, ndim=2, mode="c"] mask,
                np.ndarray[np.float32_t, ndim=1, mode="c"] crop_val_ar,
                int margin_size,
                int statbox_size):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] mask = \
            np.zeros([dims[0],dims[1]], dtype='float32')

           
    cdef np.ndarray[np.float32_t, ndim=1, mode="c"] crop_val_ar = \
            np.zeros([4], dtype='float32')            
        
    autocropper_main(&Input[0,0], &mask[0,0], &crop_val_ar[0], margin_size, statbox_size, dims[1], dims[0], 1)
    return mask


#################################################################################