/*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2019 Daniil Kazantsev

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"
#include "utils.h"

/* A method to ensure connectivity within regions of the segmented image/volume. Here we assume
 * that the MASK has been obtained using some classification/segmentation method such as k-means or gaussian
 * mixture. Some pixels/voxels have been misclassified and we check the spatial dependences
 * and correct the mask. We check the connectivity using the bresenham line algorithm within the non-local window
 * surrounding the pixel of interest.
 *
 * Input Parameters:
 * 1. MASK [0:255], the result of some classification algorithm (information-based preferably)
 * 2. The list of classes (e.g. [3,4]) to apply the method. The given order matters.
 * 3. The total number of classes in the MASK.
 * 4. The size of the Correction Window inside which the method works.

 * Output:
 * 1. MASK_upd - the UPDATED MASK where some regions have been corrected (merged) or removed
 * 2. CORRECTEDRegions - The array of the same size as MASK where all regions which were
 * changed are highlighted and the changes have been counted
 */


#ifdef __cplusplus
extern "C" {
#endif
float Mask_merge_main(unsigned char *MASK, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *SelClassesList, unsigned char *ComboClasses, int tot_combinations, int SelClassesList_length, int classesNumb, int CorrectionWindow, int iterationsNumb, int dimX, int dimY, int dimZ);
float OutiersRemoval2D(unsigned char *MASK, unsigned char *MASK_upd, long i, long j, long dimX, long dimY);
float Mask_update_main2D(unsigned char *MASK_temp, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *ClassesList, long i, long j, int CorrectionWindow, long dimX, long dimY);
float Mask_update_combo2D(unsigned char *MASK_temp, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *ClassesList, unsigned char class1, unsigned char class2, unsigned char class3, long i, long j, int CorrectionWindow, long dimX, long dimY);
int bresenham2D(int i, int j, int i1, int j1, unsigned char *MASK, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *ClassesList, int class_switcher, long dimX, long dimY);
#ifdef __cplusplus
}
#endif
