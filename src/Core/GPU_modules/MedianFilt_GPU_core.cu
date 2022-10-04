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

#include "MedianFilt_GPU_core.h"
#include "shared.h"
#include "stream_arithmetic.h"
#define MAXSTR 100
/* CUDA implementation of the median filtration and dezingering (2D/3D case) 
 * using global memory with streaming (thanks Yousef Moazzam)
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. radius_kernel: The half-size (radius) of the median filter window
 * 3. mu_threshold: if not a zero then median is applied to outliers only (zingers)

 * Output:
 * [1] Filtered or dezingered image/volume
 */

/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
template <int radius, int diameter, int midpoint> // diameter should be set to 2*radius+1
inline __device__ void medfilt_kernel_2D_global_float_t(
    float *Input,
    float *Output,
    int offset,
    int N, // faster dimension
    int M,
    int num_total,
    float mu_threshold
    )
{  
  float ValVec[diameter*diameter]; // 2D
  int i1, j1, i_m, j_m, counter = 0;

      const int i = blockDim.x * blockIdx.x + threadIdx.x;
      // calculate the number of rows to offset the j index by to get to the
      // first row of the image that the current stream should be processing
      const int j_offset = offset / N;
      const int j = blockDim.y * blockIdx.y + threadIdx.y + j_offset;
      const int index = i + N*j;

      if (index < num_total && i < N && j < M)	{
      for(i_m=-radius; i_m<=radius; i_m++) {
            i1 =  i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-radius; j_m<=radius; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
              ValVec[counter++] = Input[i1 + N*j1];
      }}
      //sort_quick(ValVec, 0, diameter*diameter); /* perform sorting */
      sort_bubble(ValVec, diameter*diameter); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midpoint]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs(Input[index] - ValVec[midpoint]) >= mu_threshold) Output[index] = ValVec[midpoint];
      }
      }
}
// instances of predefined kernel templates with selected radius/diameter
__global__ void medfilt_kernel_global_2D_r1(float *Input, float *Output, int offset, int N, int M, int num_total, float mu_threshold)
{
  medfilt_kernel_2D_global_float_t<1,3,4>(Input, Output, offset, N, M, num_total, mu_threshold);
}
__global__ void medfilt_kernel_global_2D_r2(float *Input, float *Output, int offset, int N, int M, int num_total, float mu_threshold)
{
  medfilt_kernel_2D_global_float_t<2,5,12>(Input, Output, offset, N, M, num_total, mu_threshold);
}
__global__ void medfilt_kernel_global_2D_r3(float *Input, float *Output, int offset, int N, int M, int num_total, float mu_threshold)
{
  medfilt_kernel_2D_global_float_t<3,7,24>(Input, Output, offset, N, M, num_total, mu_threshold);
}
__global__ void medfilt_kernel_global_2D_r4(float *Input, float *Output, int offset, int N, int M, int num_total, float mu_threshold)
{
  medfilt_kernel_2D_global_float_t<4,9,40>(Input, Output, offset, N, M, num_total, mu_threshold);
}
__global__ void medfilt_kernel_global_2D_r5(float *Input, float *Output, int offset, int N, int M, int num_total, float mu_threshold)
{
  medfilt_kernel_2D_global_float_t<5,11,60>(Input, Output, offset, N, M, num_total, mu_threshold);
}
