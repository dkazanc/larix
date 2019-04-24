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

cdef extern float Mask_merge_main(unsigned char *MASK, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *SelClassesList, int SelClassesList_length, int classesNumb, int CorrectionWindow, int iterationsNumb, int dimX, int dimY, int dimZ);

##############################################################################
#****************************************************************#
#********Mask (segmented image) correction module **************#
#****************************************************************#
def MASK_CORR_CPU(maskData, select_classes, total_classesNum, CorrectionWindow, iterationsNumb):
    if maskData.ndim == 2:
        return MASK_CORR_CPU_2D(maskData, select_classes, total_classesNum, CorrectionWindow, iterationsNumb)
    elif maskData.ndim == 3:
        return 0

def MASK_CORR_CPU_2D(np.ndarray[np.uint8_t, ndim=2, mode="c"] maskData,
                    np.ndarray[np.uint8_t, ndim=1, mode="c"] select_classes,
                     int total_classesNum,
                     int CorrectionWindow,
                     int iterationsNumb):

    cdef long dims[2]
    dims[0] = maskData.shape[0]
    dims[1] = maskData.shape[1]

    select_classes_length = select_classes.shape[0]

    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] mask_upd = \
            np.zeros([dims[0],dims[1]], dtype='uint8')
    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] corr_regions = \
            np.zeros([dims[0],dims[1]], dtype='uint8')

    # Run the function to process given MASK
    Mask_merge_main(&maskData[0,0], &mask_upd[0,0], &corr_regions[0,0], &select_classes[0], select_classes_length,
    total_classesNum, CorrectionWindow, iterationsNumb, dims[1], dims[0], 1)
    return (mask_upd,corr_regions)
