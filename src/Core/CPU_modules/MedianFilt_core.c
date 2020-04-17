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
 * 1. Noisy image/volume [REQUIRED]
 * 2. filter_half_window_size: The half size of the median filter window
 * 3. mu_threshold: if not a zero value then deinzger
 
 * Output:
 * [1] Filtered or dezingered image/volume
 *
 */

float medianfilter_main(float *Input, float *Output, int filter_half_window_size, float mu_threshold, int dimX, int dimY, int dimZ)
{
    int sizefilter_total;
    long i, j, index;

    /* copy input into output */
    copyIm(Input, Output, (long)(dimX), (long)(dimY), (long)(dimZ));
    
    if (dimZ <= 1) {
    /*2D case */
    sizefilter_total = (2*filter_half_window_size + 1)*(2*filter_half_window_size + 1);
    #pragma omp parallel for shared (Input, Output) private(i,  j, index)
    for(j=filter_half_window_size; j<dimY-filter_half_window_size; j++) {
      for(i=filter_half_window_size; i<dimX-filter_half_window_size; i++) {
          index = (long)(j*dimX+i);
          medfilt2D(Input, Output, filter_half_window_size, sizefilter_total, mu_threshold, i, j, index, (long)(dimX), (long)(dimY));
        }}
      }
     else {
              /*3D case*/
          }
    return *Output;
}

float medfilt2D(float *Input, float *Output, int filter_half_window_size, int sizefilter_total, float mu_threshold, long i, long j, long index, long dimX, long dimY)
{
    float *ValVec;
    long i_m, j_m, i1, j1, counter;
    int midval;
    midval = (int)(sizefilter_total*0.5f) - 1;
    ValVec = (float*) calloc(sizefilter_total, sizeof(float));

    counter = 0;
    for(i_m=-filter_half_window_size; i_m<=filter_half_window_size; i_m++) {
        i1 = i + i_m;
        for(j_m=-filter_half_window_size; j_m<=filter_half_window_size; j_m++) {
          j1 = j + j_m;
          ValVec[counter] = Input[j1*dimX+i1];
          counter++;
    }}
    //sort_bubble(ValVec, sizefilter_total); /* perform sorting */
    sort_quick(ValVec, 0, sizefilter_total); /* perform sorting */
    
    if (mu_threshold == 0.0f) {
    /* perform median filtration */
    Output[index] = ValVec[midval];
    }
    else {
    /* perform dezingering */
    if (fabs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
    else Output[index] = Input[index];
    }
    free(ValVec);
    return *Output;
}
