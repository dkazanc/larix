/*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2017 Daniil Kazantsev
Copyright 2017 Srikanth Nagella, Edoardo Pasca

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
#include "DLSDefines.h"


/* C-OMP implementation of simple inpainting shemes
 * inpainting using averaged interface values
 *
 * Input Parameters:
 * 1. Image/volume to inpaint
 * 2. Mask of the same size as (1) in 'unsigned char' format  (ones mark the region to inpaint, zeros belong to the data)
 * 3. Iterations number
 * 3. sigma - controlling parameter to start inpainting
 *
 * Output:
 * [1] Inpainted image/volume
 */


#ifdef __cplusplus
extern "C" {
#endif
DLS_EXPORT int Inpaint_simple_CPU_main(float *Input, unsigned char *Mask, float *Output, unsigned char *M_upd, int iterations, int W_halfsize, float sigma, int dimX, int dimY, int dimZ);
DLS_EXPORT void scaling_func(float *Input, unsigned char *M_upd, float *Output, float sigma, float *minmax_array, long i, long j, long k, long dimX, long dimY, long dimZ);
DLS_EXPORT void mean_inp_2D(float *Input, unsigned char *M_upd, float *Output, float sigma, int W_halfsize, long i, long j, long dimX, long dimY);
DLS_EXPORT void mean_inp_3D(float *Input, unsigned char *M_upd, float *Output, float sigma, int W_halfsize, long i, long j, long k, long dimX, long dimY, long dimZ);
#ifdef __cplusplus
}
#endif
