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
DLS_EXPORT int stripesdetect3d_main_float(float* Input, 
                           float* Output, 
                           int window_halflength_vertical,
                           int ratio_radius,
                           int ncores,
                           int dimX, int dimY, int dimZ);
DLS_EXPORT int stripesmask3d_main_float(float* Input, 
                         unsigned char* Output,
                         float threshold_val,
                         int stripe_length_min,
                         int stripe_depth_min,
                         int stripe_width_min,
                         float sensitivity,
                         int ncores, int dimX, int dimY, int dimZ);
/************3D functions ***********/
DLS_EXPORT void gradient3D_local(float *input, float *output, size_t dimX, size_t dimY, size_t dimZ, int axis, int step_size);
DLS_EXPORT void ratio_mean_stride3d(float* input, float* output,
                    int radius,
                    size_t i, size_t j, size_t k,
                    size_t dimX, size_t dimY, size_t dimZ);
DLS_EXPORT void vertical_median_stride3d(float* input, float* output,
                        int window_halflength_vertical, 
                        int window_fulllength,
                        int midval_window_index,
                        size_t i, size_t j, size_t k, size_t index,
                        size_t dimX, size_t dimY, size_t dimZ);
DLS_EXPORT void remove_inconsistent_stripes(unsigned char* mask,
                            unsigned char* out, 
                            int stripe_length_min, 
                            int stripe_depth_min, 
                            float sensitivity,
                            size_t i,
                            size_t j,
                            size_t k,
                            size_t index,
                            size_t dimX, size_t dimY, size_t dimZ);
DLS_EXPORT void merge_stripes(unsigned char* mask,
              unsigned char* out, 
              int stripe_width_min, 
              size_t i,
              size_t j,
              size_t k,
              size_t index,
              size_t dimX, size_t dimY, size_t dimZ);
#ifdef __cplusplus
}
#endif
