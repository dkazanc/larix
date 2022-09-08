/*
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
#include <stdlib.h>
#include <memory.h>
#include "stdio.h"
#include "omp.h"
#include "DLSDefines.h"

#ifndef max
    #define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
    #define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#ifdef __cplusplus
extern "C" {
#endif
DLS_EXPORT void swap(float *xp, float *yp);
DLS_EXPORT int signum(int i);
DLS_EXPORT void copyIm(float *A, float *U, long dimX, long dimY, long dimZ);
DLS_EXPORT void copyIm_unchar(unsigned char *A, unsigned char *U, int dimX, int dimY, int dimZ);
DLS_EXPORT void copyIm_unshort(unsigned short *A, unsigned short *U, int dimX, int dimY, int dimZ);
DLS_EXPORT void copyIm_roll(float *A, float *U, int dimX, int dimY, int roll_value, int switcher);
DLS_EXPORT void sort_bubble_float(float *x, int n_size);
DLS_EXPORT void sort_bubble_uint16(unsigned short *x, int n_size);
//DLS_EXPORT void sort_quick(float *x, int left_idx, int right_idx);
DLS_EXPORT void quicksort_float(float *x, int first, int last);
DLS_EXPORT void quicksort_uint16(unsigned short *x, int first, int last);
DLS_EXPORT void max_val_mask(float *Input, unsigned char *Mask, float *minmax_array, long dimX, long dimY, long dimZ);
#ifdef __cplusplus
}
#endif
