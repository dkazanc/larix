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
DLS_EXPORT int StripeWeights_main(float *input, float *output, float *grad_stats, int detectors_window_height, int detectors_window_width, int angles_window_depth, int vertical_mean_window, int gradient_gap, int ncores, long angl_size, long det_X_size, long det_Y_size);
DLS_EXPORT int StripesMergeMask_main(unsigned char *input, unsigned char *output, int stripe_width_max, int dilate, int ncores, long angl_size, long det_X_size, long det_Y_size);
/************2D functions ***********/
DLS_EXPORT void horiz_median_stride2D(float *input, float *output, int full_window_size, int midval_window_index, int detectors_window_height, int detectors_window_width, long angl_size, long det_X_size, long i, long j);
DLS_EXPORT void vert_mean_stride2D(float *input, float *output, int vertical1D_halfsize, long angl_size, long det_X_size, long i, long j);
DLS_EXPORT void stripes_merger2D(unsigned char *input, unsigned char *output, int stripe_width_max, long angl_size, long det_X_size, long i, long j);
/************3D functions ***********/
DLS_EXPORT void horiz_median_stride3D(float *input, float *output, int full_window_size, int midval_window_index, int detectors_window_height, int detectors_window_width, int angles_window_depth, long angl_size, long det_X_size, long det_Y_size, long i, long j, long k);
DLS_EXPORT void vert_mean_stride3D(float *input, float *output, int vertical1D_halfsize, long angl_size, long det_X_size, long det_Y_size, long i, long j, long k);

#ifdef __cplusplus
}
#endif
