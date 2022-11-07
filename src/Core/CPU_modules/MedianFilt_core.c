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

#include "MedianFilt_core.h"

/* C-OMP implementation of the median filtration and dezingering (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. radius: The half-size (radius) of the median filter window
 * 3. mu_threshold: if not a zero then median is applied to outliers only (zingers)

 * Output:
 * [1] Filtered or dezingered image/volume
 */

int medianfilter_main_float(float *Input, float *Output, int radius, float mu_threshold, int ncores, int dimX, int dimY, int dimZ)
{
    int sizefilter_total, diameter;
    long i, j, k, index;
    diameter = (int)(2*radius+1);
    /* copy input into output */
    copyIm(Input, Output, (long)(dimX), (long)(dimY), (long)(dimZ));

    if (ncores > 0) {
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(ncores); // Use a number of threads for all consecutive parallel regions 
    }    

    if (dimZ <= 1) {
    /*2D case */
    sizefilter_total = (int)(powf(diameter,2));
    #pragma omp parallel for shared (Input, Output) private(i, j, index)
    for(j=0; j<dimY; j++) {
      for(i=0; i<dimX; i++) {
          index = (long)(j*dimX+i);
          medfilt2D_float(Input, Output, radius, sizefilter_total, mu_threshold, i, j, index, (long)(dimX), (long)(dimY));
        }}
     } /* 2D case done */
     else {
     /* 3D case */
     sizefilter_total = (int)(powf(diameter,3));
     if (dimZ == diameter) {
     /* performs operation only on the central frame using all 3D information */
     #pragma omp parallel for shared (Input, Output) private(i, j, index)
     for(j=0; j<dimY; j++) {
       for(i=0; i<dimX; i++) {
           index = (long)((dimX*dimY)*radius + j*dimX+i);
           medfilt3D_pad_float(Input, Output, radius, sizefilter_total, mu_threshold, i, j, index, (long)(dimX), (long)(dimY));
         }}
     }
     else {
     /* Full data (traditional) 3D case */
     #pragma omp parallel for shared (Input, Output) private(i, j, k, index)
     for(k=0; k<dimZ; k++) {
       for(j=0; j<dimY; j++) {
         for(i=0; i<dimX; i++) {
           index = (long)((dimX*dimY)*k + j*dimX+i);
           medfilt3D_float(Input, Output, radius, sizefilter_total, mu_threshold, i, j, k, index, (long)(dimX), (long)(dimY), (long)(dimZ));
         }}}
     }
    } /* 3D case done */

    return 0;
}

int medianfilter_main_uint16(unsigned short *Input, unsigned short *Output, int radius, float mu_threshold, int ncores, int dimX, int dimY, int dimZ)
{
    int sizefilter_total, diameter;
    long i, j, k, index;
    diameter = (int)(2*radius+1);
    /* copy input into output */
    copyIm_unshort(Input, Output, (long)(dimX), (long)(dimY), (long)(dimZ));

    if (ncores > 0) {
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(ncores); // Use a number of threads for all consecutive parallel regions 
    }    

    if (dimZ <= 1) {
    /*2D case */
    sizefilter_total = (int)(powf(diameter,2));
    #pragma omp parallel for shared (Input, Output) private(i, j, index)
    for(j=0; j<dimY; j++) {
      for(i=0; i<dimX; i++) {
          index = (long)(j*dimX+i);
          medfilt2D_uint16(Input, Output, radius, sizefilter_total, mu_threshold, i, j, index, (long)(dimX), (long)(dimY));
        }}
     } /* 2D case done */
     else {
     /* 3D case */
     sizefilter_total = (int)(powf(diameter,3));
     if (dimZ == diameter) {
     /* performs operation only on the central frame using all 3D information */
     #pragma omp parallel for shared (Input, Output) private(i, j, index)
     for(j=0; j<dimY; j++) {
       for(i=0; i<dimX; i++) {
           index = (long)((dimX*dimY)*radius + j*dimX+i);
           medfilt3D_pad_uint16(Input, Output, radius, sizefilter_total, mu_threshold, i, j, index, (long)(dimX), (long)(dimY));
         }}
     }
     else {
     /* Full data (traditional) 3D case */
     #pragma omp parallel for shared (Input, Output) private(i, j, k, index)
     for(k=0; k<dimZ; k++) {
       for(j=0; j<dimY; j++) {
         for(i=0; i<dimX; i++) {
           index = (long)((dimX*dimY)*k + j*dimX+i);
           medfilt3D_uint16(Input, Output, radius, sizefilter_total, mu_threshold, i, j, k, index, (long)(dimX), (long)(dimY), (long)(dimZ));
         }}}
     }
    } /* 3D case done */

    return 0;
}


void medfilt2D_float(float *Input, float *Output, int radius, int sizefilter_total, float mu_threshold, long i, long j, long index, long dimX, long dimY)
{
    float *ValVec;
    long i_m, j_m, i1, j1, counter;
    int midval;
    midval = (int)(sizefilter_total/2);
    ValVec = (float*) calloc(sizefilter_total, sizeof(float));

    counter = 0l;
    for(i_m=-radius; i_m<=radius; i_m++) {
        i1 = i + i_m;
        if ((i1 < 0) || (i1 >= dimX)) i1 = i;
        for(j_m=-radius; j_m<=radius; j_m++) {
          j1 = j + j_m;
          if ((j1 < 0) || (j1 >= dimY)) j1 = j;
          ValVec[counter] = Input[j1*dimX+i1];
          counter++;
    }}
    //sort_bubble_float(ValVec, sizefilter_total); /* perform sorting */
    quicksort_float(ValVec, 0, sizefilter_total-1); /* perform sorting */

    if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
    else {
    /* perform dezingering */
    if (fabs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval]; }
    free(ValVec);
    return;
}

void medfilt2D_uint16(unsigned short *Input, unsigned short *Output, int radius, int sizefilter_total, float mu_threshold, long i, long j, long index, long dimX, long dimY)
{
    unsigned short *ValVec;
    long i_m, j_m, i1, j1, counter;
    int midval;
    midval = (int)(sizefilter_total/2);
    ValVec = (unsigned short*) calloc(sizefilter_total, sizeof(unsigned short));

    counter = 0l;
    for(i_m=-radius; i_m<=radius; i_m++) {
        i1 = i + i_m;
        if ((i1 < 0) || (i1 >= dimX)) i1 = i;
        for(j_m=-radius; j_m<=radius; j_m++) {
          j1 = j + j_m;
          if ((j1 < 0) || (j1 >= dimY)) j1 = j;
          ValVec[counter] = Input[j1*dimX+i1];
          counter++;
    }}
    //sort_bubble_uint16(ValVec, sizefilter_total); /* perform sorting */
    quicksort_uint16(ValVec, 0, sizefilter_total-1); /* perform sorting */

    if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
    else {
    /* perform dezingering */
    if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval]; }
    free(ValVec);
    return;
}


void medfilt3D_pad_float(float *Input, float *Output, int radius, int sizefilter_total, float mu_threshold, long i, long j, long index, long dimX, long dimY)
{
    float *ValVec;
    long i_m, j_m, k_m, i1, j1, counter;
    int midval;
    midval = (int)(sizefilter_total/2);
    ValVec = (float*) calloc(sizefilter_total, sizeof(float));

    counter = 0l;
    for(i_m=-radius; i_m<=radius; i_m++) {
        i1 = i + i_m;
        if ((i1 < 0) || (i1 >= dimX)) i1 = i;
        for(j_m=-radius; j_m<=radius; j_m++) {
          j1 = j + j_m;
          if ((j1 < 0) || (j1 >= dimY)) j1 = j;
          for(k_m=-radius; k_m<=radius; k_m++) {
          ValVec[counter] = Input[(dimX*dimY)*(radius + k_m) + j1*dimX+i1];
          counter++;
    }}}
    //sort_bubble_float(ValVec, sizefilter_total); /* perform bubble sort */
    quicksort_float(ValVec, 0, sizefilter_total-1); /* perform sorting */

    if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
    else {
    /* perform dezingering */
    if (fabs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval]; }
    free(ValVec);
    return;
}

void medfilt3D_pad_uint16(unsigned short *Input, unsigned short *Output, int radius, int sizefilter_total, float mu_threshold, long i, long j, long index, long dimX, long dimY)
{
    unsigned short *ValVec;
    long i_m, j_m, k_m, i1, j1, counter;
    int midval;
    midval = (int)(sizefilter_total/2);
    ValVec = (unsigned short*) calloc(sizefilter_total, sizeof(unsigned short));

    counter = 0l;
    for(i_m=-radius; i_m<=radius; i_m++) {
        i1 = i + i_m;
        if ((i1 < 0) || (i1 >= dimX)) i1 = i;
        for(j_m=-radius; j_m<=radius; j_m++) {
          j1 = j + j_m;
          if ((j1 < 0) || (j1 >= dimY)) j1 = j;
          for(k_m=-radius; k_m<=radius; k_m++) {
          ValVec[counter] = Input[(dimX*dimY)*(radius + k_m) + j1*dimX+i1];
          counter++;
    }}}
    //sort_bubble_uint16(ValVec, sizefilter_total); /* perform bubble sort */
    quicksort_uint16(ValVec, 0, sizefilter_total-1); /* perform sorting */

    if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
    else {
    /* perform dezingering */
    if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval]; }
    free(ValVec);
    return;
}

void medfilt3D_float(float *Input, float *Output, int radius, int sizefilter_total, float mu_threshold, long i, long j, long k, long index, long dimX, long dimY, long dimZ)
{
    float *ValVec;
    long i_m, j_m, k_m, i1, j1, k1, counter;
    int midval;
    midval = (int)(sizefilter_total/2);
    ValVec = (float*) calloc(sizefilter_total, sizeof(float));

    counter = 0l;
    for(i_m=-radius; i_m<=radius; i_m++) {
        i1 = i + i_m;
        if ((i1 < 0) || (i1 >= dimX)) i1 = i;
        for(j_m=-radius; j_m<=radius; j_m++) {
          j1 = j + j_m;
          if ((j1 < 0) || (j1 >= dimY)) j1 = j;
          for(k_m=-radius; k_m<=radius; k_m++) {
            k1 = k + k_m;
            if ((k1 < 0) || (k1 >= dimZ)) k1 = k;
          ValVec[counter] = Input[(dimX*dimY)*k1 + j1*dimX+i1];
          counter++;
    }}}
    //sort_bubble_float(ValVec, sizefilter_total); /* perform bubble sort */
    quicksort_float(ValVec, 0, sizefilter_total-1); /* perform sorting */

    if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
    else {
    /* perform dezingering */
    if (fabs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval]; }
    free(ValVec);
    return;
}

void medfilt3D_uint16(unsigned short *Input, unsigned short *Output, int radius, int sizefilter_total, float mu_threshold, long i, long j, long k, long index, long dimX, long dimY, long dimZ)
{
    unsigned short *ValVec;
    long i_m, j_m, k_m, i1, j1, k1, counter;
    int midval;
    midval = (int)(sizefilter_total/2);
    ValVec = (unsigned short*) calloc(sizefilter_total, sizeof(unsigned short));

    counter = 0l;
    for(i_m=-radius; i_m<=radius; i_m++) {
        i1 = i + i_m;
        if ((i1 < 0) || (i1 >= dimX)) i1 = i;
        for(j_m=-radius; j_m<=radius; j_m++) {
          j1 = j + j_m;
          if ((j1 < 0) || (j1 >= dimY)) j1 = j;
          for(k_m=-radius; k_m<=radius; k_m++) {
            k1 = k + k_m;
            if ((k1 < 0) || (k1 >= dimZ)) k1 = k;
          ValVec[counter] = Input[(dimX*dimY)*k1 + j1*dimX+i1];
          counter++;
    }}}
    //sort_bubble_uint16(ValVec, sizefilter_total); /* perform bubble sort */
    quicksort_uint16(ValVec, 0, sizefilter_total-1); /* perform sorting */

    if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
    else {
    /* perform dezingering */
    if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval]; }
    free(ValVec);
    return;
}
