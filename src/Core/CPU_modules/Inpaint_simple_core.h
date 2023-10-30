/*
# Copyright 2022 Diamond Light Source Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
DLS_EXPORT int Inpaint_simple_CPU_main(float *Input, unsigned char *Mask, float *Output, int iterations, int W_halfsize, int method_type, int ncores, int dimX, int dimY, int dimZ);
DLS_EXPORT void mean_smoothing_2D(unsigned char *Mask, float *Output, float *Updated, long i, long j, long dimX, long dimY);
DLS_EXPORT void eucl_weighting_inpainting_2D(unsigned char *M_upd, float *Output, float *Updated, float *Gauss_weights, int W_halfsize, long i, long j, long dimX, long dimY);
DLS_EXPORT void median_rand_inpainting_2D(unsigned char *M_upd, float *Output, float *Updated, int W_halfsize, int window_fullength, int method_type, long i, long j, long dimX, long dimY);
DLS_EXPORT void mean_smoothing_3D(unsigned char *Mask, float *Output, float *Updated, long i, long j, long k, long dimX, long dimY, long dimZ);
DLS_EXPORT void eucl_weighting_inpainting_3D(unsigned char *M_upd, float *Output, float *Updated, float *Gauss_weights, int W_halfsize, long i, long j, long k, long dimX, long dimY, long dimZ);
DLS_EXPORT void median_rand_inpainting_3D(unsigned char *M_upd, float *Output, float *Updated, int W_halfsize, int window_fullength, int method_type, long i, long j, long k, long dimX, long dimY, long dimZ);
#ifdef __cplusplus
}
#endif
