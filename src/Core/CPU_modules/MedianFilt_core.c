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

#include "MedianFilt_core.h"

/* C-OMP implementation of the median filtration and dezingering (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. kernel_size: The size of the median filter window
 * 3. mu_threshold: if not a zero value then deinzger

 * Output:
 * [1] Filtered or dezingered image/volume
 */

float medianfilter_main(float *Input, float *Output, int kernel_size, float mu_threshold, int dimX, int dimY, int dimZ)
{
    int sizefilter_total, kernel_half_size;
    long i, j, index;
    kernel_half_size = (int)((kernel_size-1)/2);
    printf("%i\n", kernel_half_size);
    /* copy input into output */
    copyIm(Input, Output, (long)(dimX), (long)(dimY), (long)(dimZ));

    if (dimZ <= 1) {
    /*2D case */
    sizefilter_total = (int)(pow(kernel_size,2));
    #pragma omp parallel for shared (Input, Output) private(i, j, index)
    for(j=0; j<dimY; j++) {
      for(i=0; i<dimX; i++) {
          index = (long)(j*dimX+i);
          medfilt2D(Input, Output, kernel_half_size, sizefilter_total, mu_threshold, i, j, index, (long)(dimX), (long)(dimY));
        }}
     } /* 2D case done */
     else {
     /* 3D case */
     if (dimZ == kernel_size) {
      /* provide an output of the central frame using all 3D information */
     }
     else {
     /* Full (traditional) 3D case */

     }
    } /* 3D case done */
    return *Output;
}

float medfilt2D(float *Input, float *Output, int kernel_half_size, int sizefilter_total, float mu_threshold, long i, long j, long index, long dimX, long dimY)
{
    float *ValVec;
    long i_m, j_m, i1, j1, counter;
    int midval;
    midval = (int)(sizefilter_total*0.5f) - 1;
    ValVec = (float*) calloc(sizefilter_total, sizeof(float));

    counter = 0;
    for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
        i1 = i + i_m;
        if ((i1 < 0) || (i1 >= dimX)) i1 = i;
        for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
          j1 = j + j_m;
          if ((j1 < 0) || (j1 >= dimY)) j1 = j;
          ValVec[counter] = Input[j1*dimX+i1];
          counter++;
    }}
    //sort_bubble(ValVec, sizefilter_total); /* perform sorting */
    sort_quick(ValVec, 0, sizefilter_total); /* perform sorting */

    if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
    else {
    /* perform dezingering */
    if (fabs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval]; }
    free(ValVec);
    return *Output;
}
