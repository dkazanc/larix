/* This works has been developed at Diamond Light Source Ltd.
 *
 * Copyright 2020 Daniil Kazantsev
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

/* C-OMP implementation of the median filtration and dezingering (2D/3D case)
* Input Parameters:
* 1. Noisy image/volume
* 2. kernel_size: The size of the median filter window
* 3. mu_threshold: if not a zero value then deinzger

* Output:
* [1] Filtered or dezingered image/volume
 */


#ifdef __cplusplus
extern "C" {
#endif
DLS_EXPORT int medianfilter_main(float *Input, float *Output, int kernel_size, float mu_threshold, int dimX, int dimY, int dimZ);
/************2D functions ***********/
DLS_EXPORT float medfilt2D(float *Input, float *Output, int kernel_half_size, int sizefilter_total, float mu_threshold, long i, long j, long index, long dimX, long dimY);
/************3D functions ***********/
DLS_EXPORT float medfilt3D_pad(float *Input, float *Output, int kernel_half_size, int sizefilter_total, float mu_threshold, long i, long j, long index, long dimX, long dimY);
DLS_EXPORT float medfilt3D(float *Input, float *Output, int kernel_half_size, int sizefilter_total, float mu_threshold, long i, long j, long k, long index, long dimX, long dimY, long dimZ);

#ifdef __cplusplus
}
#endif
