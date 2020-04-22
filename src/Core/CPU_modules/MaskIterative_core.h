/*
 *
 * Copyright 2019 Daniil Kazantsev
  *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"
#include "utils.h"
#include "DLSDefines.h"

#ifdef __cplusplus
extern "C" {
#endif
DLS_EXPORT float MASK_evolve_main(float *Input, unsigned char *MASK_in, unsigned char *MASK_out, float threhsold, int iterations, int connectivity, float value1, float value2, int dimX, int dimY, int dimZ);
DLS_EXPORT float MASK_evolve_conditional_main(float *Input, unsigned char *MASK_in, unsigned char *MASK_conditional, unsigned char *MASK_out, float threhsold, int iterations, int connectivity, float value1, float value2, int dimX, int dimY, int dimZ);
/************2D functions ***********/
DLS_EXPORT float mask_region_MADmean(float *Input, unsigned char *MASK, float *maskreg_value, long dimX, long dimY);
DLS_EXPORT float mask_region_MADmedian(float *Input, unsigned char *MASK, float *maskreg_value, long dimX, long dimY);
DLS_EXPORT float mask_update4(float *Input, unsigned char *MASK, float *maskreg_value, float threhsold, long dimX, long dimY);
DLS_EXPORT float mask_update8(float *Input, unsigned char *MASK, float *maskreg_value, float threhsold,long dimX, long dimY);
DLS_EXPORT float mask_update_con4(float *Input, unsigned char *MASK_conditional, unsigned char *MASK, float *maskreg_value, float threhsold, long dimX, long dimY);
DLS_EXPORT float mask_update_con8(float *Input, unsigned char *MASK_conditional, unsigned char *MASK, float *maskreg_value, float threhsold, long dimX, long dimY);
/************3D functions ***********/
DLS_EXPORT float mask_region_MADmean3D(float *Input, unsigned char *MASK, float *maskreg_value, long dimX, long dimY, long dimZ);
DLS_EXPORT float mask_region_MADmedian3D(float *Input, unsigned char *MASK, float *maskreg_value, long dimX, long dimY, long dimZ);
DLS_EXPORT float mask_update3D_4(float *Input, unsigned char *MASK, float *maskreg_value, float threhsold, long dimX, long dimY, long dimZ);
DLS_EXPORT float mask_update3D_8(float *Input, unsigned char *MASK, float *maskreg_value, float threhsold, long dimX, long dimY, long dimZ);
DLS_EXPORT float mask_update3D_6(float *Input, unsigned char *MASK, float *maskreg_value, float threhsold, long dimX, long dimY, long dimZ);
DLS_EXPORT float mask_update3D_26(float *Input, unsigned char *MASK, float *maskreg_value, float threhsold,  long dimX, long dimY, long dimZ);
DLS_EXPORT float mask_update_con3D_4(float *Input, unsigned char *MASK_conditional, unsigned char *MASK, float *maskreg_value, float threhsold, long dimX, long dimY, long dimZ);
DLS_EXPORT float mask_update_con3D_8(float *Input, unsigned char *MASK_conditional, unsigned char *MASK, float *maskreg_value, float threhsold, long dimX, long dimY, long dimZ);
DLS_EXPORT float mask_update_con3D_6(float *Input, unsigned char *MASK_conditional, unsigned char *MASK, float *maskreg_value, float threhsold, long dimX, long dimY, long dimZ);
DLS_EXPORT float mask_update_con3D_26(float *Input, unsigned char *MASK_conditional, unsigned char *MASK, float *maskreg_value, float threhsold, long dimX, long dimY, long dimZ);
#ifdef __cplusplus
}
#endif
