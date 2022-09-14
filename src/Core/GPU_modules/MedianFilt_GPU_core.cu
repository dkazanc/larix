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
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. kernel_size: The size of the median filter window
 * 3. mu_threshold: if not a zero value then deinzger

 * Output:
 * [1] Filtered or dezingered image/volume
 */
/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/

__global__ void kernel(float *Input, float* Output, int offset,  int N, int M, int num_total)
  {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int j = blockDim.y * blockIdx.y + threadIdx.y;
  const int index = offset + i + N*j;

      if (index < num_total) Output[index] = Input[index] * 100;
  }

__global__ void medfilt1_kernel_2D(float *Input, float* Output, int offset, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int num_total)
  {
      float ValVec[CONSTVECSIZE_9];
      int i1, j1, i_m, j_m, counter = 0;

      const int i = blockDim.x * blockIdx.x + threadIdx.x;
      // calculate the number of rows to offset the j index by to get to the
      // first row of the image that the current stream should be processing
      const int j_offset = offset / N;
      const int j = blockDim.y * blockIdx.y + threadIdx.y + j_offset;
      const int index = i + N*j;

      if (index < num_total && i < N && j < M)	{
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 =  i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
              ValVec[counter++] = Input[i1 + N*j1];
      }}
      //sort_quick(ValVec, 0, CONSTVECSIZE_9); /* perform sorting */
      sort_bubble(ValVec, CONSTVECSIZE_9); /* perform sorting */

      if (mu_threshold == 0.0f) {
            Output[index] = ValVec[midval]; /* perform median filtration */
        }
      else {
      /* perform dezingering */
      if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
      }
      }
  }

__global__ void medfilt2_kernel_2D(float *Input, float* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int num_total)
    {
        float ValVec[CONSTVECSIZE_25];
        int i1, j1, i_m, j_m, counter = 0;

        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        const int index = i + N*j;

        if (index < num_total)	{
        for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
              i1 = i + i_m;
              if ((i1 < 0) || (i1 >= N)) i1 = i;
              for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
                j1 = j + j_m;
                if ((j1 < 0) || (j1 >= M)) j1 = j;
                ValVec[counter++] = Input[i1 + N*j1];
        }}
        //sort_quick(ValVec, 0, CONSTVECSIZE_25); /* perform sorting */
        sort_bubble(ValVec, CONSTVECSIZE_25); /* perform sorting */
        if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
        else {
        /* perform dezingering */
        if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
          }
        }
    }

__global__ void medfilt3_kernel_2D(float *Input, float* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int num_total)
    {
        float ValVec[CONSTVECSIZE_49];
        int i1, j1, i_m, j_m, counter = 0;

        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        const int index = i + N*j;

        if (index < num_total)	{
        for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
              i1 = i + i_m;
              if ((i1 < 0) || (i1 >= N)) i1 = i;
              for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
                j1 = j + j_m;
                if ((j1 < 0) || (j1 >= M)) j1 = j;
                ValVec[counter++] = Input[i1 + N*j1];
        }}
        //sort_quick(ValVec, 0, CONSTVECSIZE_49); /* perform sorting */
        sort_bubble(ValVec, CONSTVECSIZE_49); /* perform sorting */
        if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
        else {
        /* perform dezingering */
        if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
          }
        }
    }

__global__ void medfilt4_kernel_2D(float *Input, float* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int num_total)
    {
        float ValVec[CONSTVECSIZE_81];
        int i1, j1, i_m, j_m, counter = 0;

        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        const int index = i + N*j;

        if (index < num_total)	{
        for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
              i1 = i + i_m;
              if ((i1 < 0) || (i1 >= N)) i1 = i;
              for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
                j1 = j + j_m;
                if ((j1 < 0) || (j1 >= M)) j1 = j;
                ValVec[counter++] = Input[i1 + N*j1];
        }}
        //sort_quick(ValVec, 0, CONSTVECSIZE_81); /* perform sorting */
        sort_bubble(ValVec, CONSTVECSIZE_81); /* perform sorting */
        if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
        else {
        /* perform dezingering */
        if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
          }
        }
    }

__global__ void medfilt5_kernel_2D(float *Input, float* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int num_total)
    {
        float ValVec[CONSTVECSIZE_121];
        int i1, j1, i_m, j_m, counter = 0;

        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        const int index = i + N*j;

        if (index < num_total)	{
        for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
              i1 = i + i_m;
              if ((i1 < 0) || (i1 >= N)) i1 = i;
              for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
                j1 = j + j_m;
                if ((j1 < 0) || (j1 >= M)) j1 = j;
                ValVec[counter++] = Input[i1 + N*j1];
        }}
        //sort_quick(ValVec, 0, CONSTVECSIZE_121); /* perform sorting */
        sort_bubble(ValVec, CONSTVECSIZE_121); /* perform sorting */
        if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
        else {
        /* perform dezingering */
        if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
          }
        }
    }

__global__ void medfilt1_kernel_uint16_2D(unsigned short *Input, unsigned short* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int num_total)
  {
      unsigned short ValVec[CONSTVECSIZE_9];
      int i1, j1, i_m, j_m, counter = 0;

      const int i = blockDim.x * blockIdx.x + threadIdx.x;
      const int j = blockDim.y * blockIdx.y + threadIdx.y;
      const int index = i + N*j;

      if (index < num_total)	{
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
              ValVec[counter++] = Input[i1 + N*j1];
      }}
      //sort_quick(ValVec, 0, CONSTVECSIZE_9); /* perform sorting */
      sort_bubble_uint16(ValVec, CONSTVECSIZE_9); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval];
      }
      }
  }

__global__ void medfilt2_kernel_uint16_2D(unsigned short *Input, unsigned short* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int num_total)
    {
        unsigned short ValVec[CONSTVECSIZE_25];
        int i1, j1, i_m, j_m, counter = 0;

        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        const int index = i + N*j;

        if (index < num_total)	{
        for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
              i1 = i + i_m;
              if ((i1 < 0) || (i1 >= N)) i1 = i;
              for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
                j1 = j + j_m;
                if ((j1 < 0) || (j1 >= M)) j1 = j;
                ValVec[counter++] = Input[i1 + N*j1];
        }}
        //sort_quick(ValVec, 0, CONSTVECSIZE_25); /* perform sorting */
        sort_bubble_uint16(ValVec, CONSTVECSIZE_25); /* perform sorting */
        if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
        else {
        /* perform dezingering */
        if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval];
          }
        }
    }

__global__ void medfilt3_kernel_uint16_2D(unsigned short *Input, unsigned short* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int num_total)
    {
        unsigned short ValVec[CONSTVECSIZE_49];
        int i1, j1, i_m, j_m, counter = 0;

        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        const int index = i + N*j;

        if (index < num_total)	{
        for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
              i1 = i + i_m;
              if ((i1 < 0) || (i1 >= N)) i1 = i;
              for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
                j1 = j + j_m;
                if ((j1 < 0) || (j1 >= M)) j1 = j;
                ValVec[counter++] = Input[i1 + N*j1];
        }}
        //sort_quick(ValVec, 0, CONSTVECSIZE_49); /* perform sorting */
        sort_bubble_uint16(ValVec, CONSTVECSIZE_49); /* perform sorting */
        if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
        else {
        /* perform dezingering */
        if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval];
          }
        }
    }

__global__ void medfilt4_kernel_uint16_2D(unsigned short *Input, unsigned short* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int num_total)
    {
        unsigned short ValVec[CONSTVECSIZE_81];
        int i1, j1, i_m, j_m, counter = 0;

        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        const int index = i + N*j;

        if (index < num_total)	{
        for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
              i1 = i + i_m;
              if ((i1 < 0) || (i1 >= N)) i1 = i;
              for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
                j1 = j + j_m;
                if ((j1 < 0) || (j1 >= M)) j1 = j;
                ValVec[counter++] = Input[i1 + N*j1];
        }}
        //sort_quick(ValVec, 0, CONSTVECSIZE_81); /* perform sorting */
        sort_bubble_uint16(ValVec, CONSTVECSIZE_81); /* perform sorting */
        if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
        else {
        /* perform dezingering */
        if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval];
          }
        }
    }

__global__ void medfilt5_kernel_uint16_2D(unsigned short *Input, unsigned short* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int num_total)
    {
        unsigned short ValVec[CONSTVECSIZE_121];
        int i1, j1, i_m, j_m, counter = 0;

        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        const int index = i + N*j;

        if (index < num_total)	{
        for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
              i1 = i + i_m;
              if ((i1 < 0) || (i1 >= N)) i1 = i;
              for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
                j1 = j + j_m;
                if ((j1 < 0) || (j1 >= M)) j1 = j;
                ValVec[counter++] = Input[i1 + N*j1];
        }}
        //sort_quick(ValVec, 0, CONSTVECSIZE_121); /* perform sorting */
        sort_bubble_uint16(ValVec, CONSTVECSIZE_121); /* perform sorting */
        if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
        else {
        /* perform dezingering */
        if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval];
          }
        }
    }

/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/
__global__ void medfilt1_kernel_3D(float *Input, float* Output, unsigned long long offset, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
  {
      float ValVec[CONSTVECSIZE_27];
      long i1, j1, k1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      // calculate the number of vertical slices to offset the k index by to get
      // to the first vertical slice of the volume that the current stream
      // should be processing
      const unsigned long long k_offset = offset / ((unsigned long long)N*(unsigned long long)M);
      const unsigned long long k = (unsigned long long)blockDim.z * (unsigned long long)blockIdx.z + (unsigned long long)threadIdx.z + (unsigned long long)k_offset;
      const unsigned long long index = (unsigned long long)i + (unsigned long long)N*(unsigned long long)j + (unsigned long long)N*(unsigned long long)M*(unsigned long long)k;

      if (index < num_total && i < N && j < M && k < Z)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  k1 = k + k_m;
                  if ((k1 < 0) || (k1 >= Z)) k1 = k;
                  ValVec[counter] = Input[i1 + N*j1 + N*M*k1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_27); /* perform sorting */
      sort_bubble(ValVec, CONSTVECSIZE_27); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
        }
      }
      return;
  }

__global__ void medfilt2_kernel_3D(float *Input, float* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
    {
      float ValVec[CONSTVECSIZE_125];
      long i1, j1, k1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long k = blockDim.z * blockIdx.z + threadIdx.z;
      const long index = N*M*k + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  k1 = k + k_m;
                  if ((k1 < 0) || (k1 >= Z)) k1 = k;
                  ValVec[counter] = Input[N*M*k1 + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_125); /* perform sorting */
      sort_bubble(ValVec, CONSTVECSIZE_125); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
        }
      }
      return;
    }

__global__ void medfilt3_kernel_3D(float *Input, float* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
    {
      float ValVec[CONSTVECSIZE_343];
      long i1, j1, k1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long k = blockDim.z * blockIdx.z + threadIdx.z;
      const long index = N*M*k + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  k1 = k + k_m;
                  if ((k1 < 0) || (k1 >= Z)) k1 = k;
                  ValVec[counter] = Input[N*M*k1 + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_343); /* perform sorting */
      sort_bubble(ValVec, CONSTVECSIZE_343); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
        }
      }
      return;
    }

__global__ void medfilt4_kernel_3D(float *Input, float* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
    {
      float ValVec[CONSTVECSIZE_729];
      long i1, j1, k1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long k = blockDim.z * blockIdx.z + threadIdx.z;
      const long index = N*M*k + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  k1 = k + k_m;
                  if ((k1 < 0) || (k1 >= Z)) k1 = k;
                  ValVec[counter] = Input[N*M*k1 + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_729); /* perform sorting */
      sort_bubble(ValVec, CONSTVECSIZE_729); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
        }
      }
      return;
    }

__global__ void medfilt5_kernel_3D(float *Input, float* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
    {
      float ValVec[CONSTVECSIZE_1331];
      long i1, j1, k1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long k = blockDim.z * blockIdx.z + threadIdx.z;
      const long index = N*M*k + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  k1 = k + k_m;
                  if ((k1 < 0) || (k1 >= Z)) k1 = k;
                  ValVec[counter] = Input[N*M*k1 + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_1331); /* perform sorting */
      sort_bubble(ValVec, CONSTVECSIZE_1331); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
        }
      }
      return;
    }

__global__ void medfilt1_pad_kernel_3D(float *Input, float* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
  {
      float ValVec[CONSTVECSIZE_27];
      long i1, j1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long index = N*M*kernel_half_size + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  ValVec[counter] = Input[N*M*(kernel_half_size + k_m) + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_27); /* perform sorting */
      sort_bubble(ValVec, CONSTVECSIZE_27); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];  }
      }
      return;
  }

__global__ void medfilt2_pad_kernel_3D(float *Input, float* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
    {
      float ValVec[CONSTVECSIZE_125];
      long i1, j1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long index = N*M*kernel_half_size + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  ValVec[counter] = Input[N*M*(kernel_half_size + k_m) + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_125); /* perform sorting */
      sort_bubble(ValVec, CONSTVECSIZE_125); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
        }
      }
      return;
    }

__global__ void medfilt3_pad_kernel_3D(float *Input, float* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
    {
      float ValVec[CONSTVECSIZE_343];
      long i1, j1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long index = N*M*kernel_half_size + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  ValVec[counter] = Input[N*M*(kernel_half_size + k_m) + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_343); /* perform sorting */
      sort_bubble(ValVec, CONSTVECSIZE_343); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
        }
      }
      return;
    }

__global__ void medfilt4_pad_kernel_3D(float *Input, float* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
    {
      float ValVec[CONSTVECSIZE_729];
      long i1, j1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long index = N*M*kernel_half_size + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  ValVec[counter] = Input[N*M*(kernel_half_size + k_m) + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_729); /* perform sorting */
      sort_bubble(ValVec, CONSTVECSIZE_729); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
        }
      }
      return;
    }

__global__ void medfilt5_pad_kernel_3D(float *Input, float* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
    {
      float ValVec[CONSTVECSIZE_1331];
      long i1, j1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long index = N*M*kernel_half_size + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  ValVec[counter] = Input[N*M*(kernel_half_size + k_m) + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_1331); /* perform sorting */
      sort_bubble(ValVec, CONSTVECSIZE_1331); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
        }
      }
      return;
    }
/* ======================================================================= */

__global__ void medfilt1_pad_kernel_uint16_3D(unsigned short *Input, unsigned short* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
  {
      unsigned short ValVec[CONSTVECSIZE_27];
      long i1, j1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long index = N*M*kernel_half_size + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  ValVec[counter] = Input[N*M*(kernel_half_size + k_m) + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_27); /* perform sorting */
      sort_bubble_uint16(ValVec, CONSTVECSIZE_27); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval];}
      }
      return;
  }

__global__ void medfilt2_pad_kernel_uint16_3D(unsigned short *Input, unsigned short* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
    {
      unsigned short ValVec[CONSTVECSIZE_125];
      long i1, j1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long index = N*M*kernel_half_size + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  ValVec[counter] = Input[N*M*(kernel_half_size + k_m) + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_125); /* perform sorting */
      sort_bubble_uint16(ValVec, CONSTVECSIZE_125); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval];
        }
      }
      return;
    }

__global__ void medfilt3_pad_kernel_uint16_3D(unsigned short *Input, unsigned short* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
    {
      unsigned short ValVec[CONSTVECSIZE_343];
      long i1, j1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long index = N*M*kernel_half_size + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  ValVec[counter] = Input[N*M*(kernel_half_size + k_m) + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_343); /* perform sorting */
      sort_bubble_uint16(ValVec, CONSTVECSIZE_343); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval];
        }
      }
      return;
    }

__global__ void medfilt4_pad_kernel_uint16_3D(unsigned short *Input, unsigned short *Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
    {
      unsigned short ValVec[CONSTVECSIZE_729];
      long i1, j1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long index = N*M*kernel_half_size + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  ValVec[counter] = Input[N*M*(kernel_half_size + k_m) + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_729); /* perform sorting */
      sort_bubble_uint16(ValVec, CONSTVECSIZE_729); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval];
        }
      }
      return;
    }

__global__ void medfilt5_pad_kernel_uint16_3D(unsigned short *Input, unsigned short *Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
    {
      unsigned short ValVec[CONSTVECSIZE_1331];
      long i1, j1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long index = N*M*kernel_half_size + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  ValVec[counter] = Input[N*M*(kernel_half_size + k_m) + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_1331); /* perform sorting */
      sort_bubble_uint16(ValVec, CONSTVECSIZE_1331); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval];
        }
      }
      return;
    }

__global__ void medfilt1_kernel_uint16_3D(unsigned short *Input, unsigned short *Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
  {
      unsigned short ValVec[CONSTVECSIZE_27];
      long i1, j1, k1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long k = blockDim.z * blockIdx.z + threadIdx.z;
      const long index = N*M*k + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  k1 = k + k_m;
                  if ((k1 < 0) || (k1 >= Z)) k1 = k;
                  ValVec[counter] = Input[N*M*k1 + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_27); /* perform sorting */
      sort_bubble_uint16(ValVec, CONSTVECSIZE_27); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval];
        }
      }
      return;
    }

__global__ void medfilt2_kernel_uint16_3D(unsigned short *Input, unsigned short *Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
    {
      unsigned short ValVec[CONSTVECSIZE_125];
      long i1, j1, k1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long k = blockDim.z * blockIdx.z + threadIdx.z;
      const long index = N*M*k + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  k1 = k + k_m;
                  if ((k1 < 0) || (k1 >= Z)) k1 = k;
                  ValVec[counter] = Input[N*M*k1 + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_125); /* perform sorting */
      sort_bubble_uint16(ValVec, CONSTVECSIZE_125); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval];
        }
      }
      return;
    }

__global__ void medfilt3_kernel_uint16_3D(unsigned short *Input, unsigned short *Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
    {
      unsigned short ValVec[CONSTVECSIZE_343];
      long i1, j1, k1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long k = blockDim.z * blockIdx.z + threadIdx.z;
      const long index = N*M*k + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  k1 = k + k_m;
                  if ((k1 < 0) || (k1 >= Z)) k1 = k;
                  ValVec[counter] = Input[N*M*k1 + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_343); /* perform sorting */
      sort_bubble_uint16(ValVec, CONSTVECSIZE_343); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval];
        }
      }
      return;
    }

__global__ void medfilt4_kernel_uint16_3D(unsigned short *Input, unsigned short *Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
    {
      unsigned short ValVec[CONSTVECSIZE_729];
      long i1, j1, k1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long k = blockDim.z * blockIdx.z + threadIdx.z;
      const long index = N*M*k + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  k1 = k + k_m;
                  if ((k1 < 0) || (k1 >= Z)) k1 = k;
                  ValVec[counter] = Input[N*M*k1 + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_729); /* perform sorting */
      sort_bubble_uint16(ValVec, CONSTVECSIZE_729); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval];
        }
      }
      return;
    }

__global__ void medfilt5_kernel_uint16_3D(unsigned short *Input, unsigned short *Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
    {
      unsigned short ValVec[CONSTVECSIZE_1331];
      long i1, j1, k1, i_m, j_m, k_m, counter;

      const long i = blockDim.x * blockIdx.x + threadIdx.x;
      const long j = blockDim.y * blockIdx.y + threadIdx.y;
      const long k = blockDim.z * blockIdx.z + threadIdx.z;
      const long index = N*M*k + i + N*j;

      if (index < num_total)	{
      counter = 0l;
      for(i_m=-kernel_half_size; i_m<=kernel_half_size; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-kernel_half_size; j_m<=kernel_half_size; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-kernel_half_size; k_m<=kernel_half_size; k_m++) {
                  k1 = k + k_m;
                  if ((k1 < 0) || (k1 >= Z)) k1 = k;
                  ValVec[counter] = Input[N*M*k1 + i1 + N*j1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, CONSTVECSIZE_1331); /* perform sorting */
      sort_bubble_uint16(ValVec, CONSTVECSIZE_1331); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs((int)(Input[index]) - (int)(ValVec[midval])) >= (int)(mu_threshold)) Output[index] = ValVec[midval];
        }
      }
      return;
    }
/****************************************************************************/


/////////////////////////////////////////////////
/////////////// HOST FUNCTION ///////////////////
/////////////////////////////////////////////////
extern "C" int MedianFilt_GPU_main_float32(float *Input, float *Output, int kernel_size, float mu_threshold, int gpu_device, int N, int M, int Z)
{
  int deviceCount = -1; // number of devices
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
      fprintf(stderr, "No CUDA devices found\n");
       return -1;
   }

        int sizefilter_total, kernel_half_size, midval;
        const unsigned long long ImSize = (unsigned long long)N*(unsigned long long)M*(unsigned long long)Z;
        float *d_input0, *d_output0;

        /*set GPU device*/
        checkCudaErrors(cudaSetDevice(gpu_device));

        const int nStreams = 4;
        const unsigned long long n = ImSize;
        const unsigned long long bytes = n * sizeof(float);

        // create events and streams
        cudaStream_t stream[nStreams];
        for (int i = 0; i < nStreams; ++i)
          checkCudaErrors( cudaStreamCreate(&stream[i]) );

        // allocate memory on the device
        checkCudaErrors(cudaMalloc((void**)&d_input0, bytes));
        checkCudaErrors(cudaMalloc((void**)&d_output0, bytes));

        // allocate pinned memory for data
        float* pinned_mem_input;
        float* pinned_mem_output;
        cudaHostAlloc(&pinned_mem_input, bytes, cudaHostAllocDefault);
        for (unsigned long long i = 0; i < ImSize; i++) {
          pinned_mem_input[i] = Input[i];
        }
        cudaHostAlloc(&pinned_mem_output, bytes, cudaHostAllocDefault);

        /*
        checkCudaErrors(cudaMemcpy(d_input0,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_output0,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
        */
	if (Z == 1) {
        /*2D case */
        //dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
        //dim3 dimGrid(idivup((N/nStreams),BLKXSIZE2D), idivup((M/nStreams),BLKYSIZE2D));

        const StreamInfo streamInfo = calculate_stream_size(nStreams, N, M, Z);
        const unsigned long long stream_bytes = streamInfo.stream_size * sizeof(float);
        const int leftover_rows = M - ((nStreams - 1) * streamInfo.slices_per_stream);
        const unsigned long long leftover_row_bytes = leftover_rows * (unsigned long long)N * sizeof(float);
        // calculate the number of bytes in a single row of the image
        const unsigned long long im_row_bytes = (unsigned long long)N * sizeof(float);

        const int blockSize = 16;

        // Each stream will process a subset of the data with shape
        // (rows_per_stream, N). Calculate a (reasonably) optimal grid size of
        // (16, 16) thread-blocks to map onto the 2D image
        int grid_x_dim = 0;
        int grid_y_dim = 0;
        int vertical_count = 0;
        int horizontal_count = 0;

        while (vertical_count < streamInfo.slices_per_stream) {
          grid_y_dim++;
          vertical_count += blockSize;
        }

        while (horizontal_count < N) {
          grid_x_dim++;
          horizontal_count += blockSize;
        }

        dim3 dimBlock(blockSize,blockSize);
        dim3 dimGrid(grid_x_dim, grid_y_dim);

        sizefilter_total = (int)(pow(kernel_size,2));
        kernel_half_size = (int)((kernel_size-1)/2);
        midval = (int)(sizefilter_total/2);

        // number of bytes to copy for a stream
        int bytes_to_copy_h2d;
        int bytes_to_copy_d2h;
        int copy_offset_h2d;
        int copy_offset_d2h;
        int process_offset;
        for (int i = 0; i < nStreams; ++i) {
          // every stream should still be processing the same number of
          // elements, just that the memeory address where the data for a
          // particular stream starts changes for every stream
          process_offset = i * streamInfo.stream_size;
          // since every stream processes the same number of elements, it should
          // always copy the same number of elements back from GPU to CPU, just
          // with a different memory address offset for each stream
          copy_offset_d2h = i * streamInfo.stream_size;
          bytes_to_copy_d2h = stream_bytes;
          /* copy streamed data from host to the device */
          if (nStreams > 1) {
            if (i == 0) {
              // for stream 0, copy an extra row of pixels that comes after the
              // rows being processed in this stream, so then those neighbouring
              // pixels are avaiable for processing in this stream
              bytes_to_copy_h2d = stream_bytes + im_row_bytes;
              copy_offset_h2d = 0;
            }
            else if (1 <= i && i < nStreams - 1) {
              // for streams 1 to (nStreams - 1), copy an extra row of pixels in
              // the image that come before AND after the rows being processed
              // in these streams over to the GPU:
              // this so then neighbouring pixels of the first and last rows of
              // the rows being processed in this stream are guaranteed to be in
              // GPU memory
              copy_offset_h2d = i * streamInfo.stream_size - N;
              bytes_to_copy_h2d = stream_bytes + 2*im_row_bytes;
            }
            else {
              // for the last stream, copy an extra row of pixels that comes
              // before the rows being processed in this stream, so then those
              // neighbouring pixels are available in this stream
              copy_offset_h2d = i * streamInfo.stream_size - N;
              bytes_to_copy_h2d = leftover_row_bytes + im_row_bytes;
              bytes_to_copy_d2h = leftover_rows * N * sizeof(float);
            }
          }
          else {
            copy_offset_h2d = 0;
            bytes_to_copy_h2d = stream_bytes;
          }
          checkCudaErrors( cudaMemcpyAsync(&d_input0[copy_offset_h2d], &pinned_mem_input[copy_offset_h2d],
                                     bytes_to_copy_h2d, cudaMemcpyHostToDevice,
                                     stream[i]) );
          // running the kernel
          //kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
          //medfilt1_kernel_2D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
          //medfilt1_kernel_2D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
          //medfilt1_kernel_2D<<<dimGrid, dimBlock, 0, stream[i] >>>(d_input0, d_output0, offset, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
          //medfilt1_kernel_2D<<<numOfBlocks, numOfThreadsPerBlocks, 0, stream[i] >>>(d_input0+offset, d_output0+offset, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
          //simple_kernel<<<dimGrid, dimBlock, 0, stream[i] >>>(d_input0+offset, d_output0+offset, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
          //kernel<<<dimGrid, dimBlock, 0, stream[i]>>>(d_input0, d_output0, offset, N, M, ImSize);
          medfilt1_kernel_2D<<<dimGrid, dimBlock, 0, stream[i] >>>(d_input0, d_output0, process_offset, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);

          /* copy processed data from device to the host */
          checkCudaErrors( cudaMemcpyAsync(&pinned_mem_output[copy_offset_d2h], &d_output0[copy_offset_d2h],
                                     bytes_to_copy_d2h, cudaMemcpyDeviceToHost,
                                     stream[i]) );
          checkCudaErrors( cudaDeviceSynchronize() );
          }

        /*
        if (kernel_size == 3) medfilt1_kernel_2D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
        else if (kernel_size == 5) medfilt2_kernel_2D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
        else if (kernel_size == 7) medfilt3_kernel_2D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
        else if (kernel_size == 9) medfilt4_kernel_2D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
        else medfilt5_kernel_2D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
        */
        //checkCudaErrors( cudaDeviceSynchronize() );
        //checkCudaErrors(cudaPeekAtLastError() );
       }
	else {
		    /*3D case*/

        // note that the slabs of the volume that are being processed in each
        // stream are VERTICAL slabs rather than HORIZONTAL slabs, due to how
        // the ordering of the test volume data is:
        // - x index increases the quickest
        // - y index increases next
        // - z index increases the slowest

        const StreamInfo streamInfo = calculate_stream_size(nStreams, N, M, Z);
        const unsigned long long stream_bytes = streamInfo.stream_size * sizeof(float);
        const int leftover_slices = Z - ((nStreams - 1) * streamInfo.slices_per_stream);
        const unsigned long long leftover_slab_bytes = leftover_slices * (unsigned long long)N * (unsigned long long)M * sizeof(float);
        // calculate the number of bytes in a single slice of the volume
        const unsigned long long vol_slice_bytes = (unsigned long long)N * (unsigned long long)M * sizeof(float);

        // Each stream will process a subset of the data with shape
        // (x, y, z) = (N, M, slices_per_stream). Calculate a (reasonably)
        // optimal grid size of (8, 8, 8) thread-blocks to map onto the 3D
        // volume
        int grid_x_dim = 0;
        int grid_y_dim = 0;
        int grid_z_dim = 0;
        int vertical_count = 0;
        int horizontal_count = 0;
        int depth_count = 0;

        while (vertical_count < M) {
          grid_y_dim++;
          vertical_count += BLKYSIZE;
        }

        while (horizontal_count < N) {
          grid_x_dim++;
          horizontal_count += BLKXSIZE;
        }

        while (depth_count < streamInfo.slices_per_stream) {
          grid_z_dim++;
          depth_count += BLKZSIZE;
        }

        dim3 dimBlock(BLKXSIZE,BLKYSIZE,BLKZSIZE);
        dim3 dimGrid(grid_x_dim, grid_y_dim, grid_z_dim);
        sizefilter_total = (int)(pow(kernel_size, 3));
        kernel_half_size = (int)((kernel_size-1)/2);
        midval = (int)(sizefilter_total/2);


        if (Z == kernel_size) {
        /* performs operation only on the central frame using all 3D information */
        /*
        if (kernel_size == 3) medfilt1_pad_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        else if (kernel_size == 5) medfilt2_pad_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        else if (kernel_size == 7) medfilt3_pad_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        else if (kernel_size == 9) medfilt4_pad_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        else medfilt5_pad_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        */
        }
        else {

          // Benchmark approach 1: WITHOUT streaming
          int default_stream_grid_x_dim;
          int default_stream_grid_y_dim;
          int default_stream_grid_z_dim;
          int default_stream_vertical_count = 0;
          int default_stream_horizontal_count = 0;
          int default_stream_depth_count = 0;
          int default_stream_slices_per_stream = ceil((float)n/(float)(N*M));

          while (default_stream_vertical_count < M) {
            default_stream_grid_y_dim++;
            default_stream_vertical_count += BLKXSIZE;
          }

          while (default_stream_horizontal_count < N) {
            default_stream_grid_x_dim++;
            default_stream_horizontal_count += BLKYSIZE;
          }

          while (default_stream_depth_count < default_stream_slices_per_stream) {
            default_stream_grid_z_dim++;
            default_stream_depth_count += BLKZSIZE;
          }

          dim3 defaultStreamDimBlock(BLKXSIZE, BLKYSIZE, BLKZSIZE);
          dim3 defaultStreamDimGrid(default_stream_grid_x_dim,
            default_stream_grid_y_dim, default_stream_grid_z_dim);
          cudaEvent_t start_without_streaming, end_without_streaming;
          float elapsed_time_without_streaming;
          cudaEventCreate(&start_without_streaming);
          cudaEventCreate(&end_without_streaming);
          cudaEventRecord(start_without_streaming);

          cudaMemcpy(d_input0, Input, bytes, cudaMemcpyHostToDevice);

          /* Full data (traditional) 3D case */
          if (kernel_size == 3) medfilt1_kernel_3D<<<defaultStreamDimGrid,defaultStreamDimBlock>>>(d_input0, d_output0, 0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
          //if (kernel_size == 3) medfilt1_kernel_3D<<<dimGrid,dimBlock, 0, stream[0]>>>(d_input0, d_output0, 0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
          else if (kernel_size == 5) medfilt2_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
          else if (kernel_size == 7) medfilt3_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
          else if (kernel_size == 9) medfilt4_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
          else medfilt5_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);

          cudaMemcpy(Output, d_output0, bytes, cudaMemcpyDeviceToHost);

          // finish clocking approach 1: WITHOUT streaming
          cudaEventRecord(end_without_streaming);
          cudaEventSynchronize(end_without_streaming);
          cudaEventElapsedTime(&elapsed_time_without_streaming,
            start_without_streaming, end_without_streaming);
          printf("3D median filter WITHOUT streaming (ie, just using the single default stream) took %f ms\n",
            elapsed_time_without_streaming);
          cudaEventDestroy(start_without_streaming);
          cudaEventDestroy(end_without_streaming);


          // number of bytes to copy for a stream
          unsigned long long bytes_to_copy_h2d;
          unsigned long long bytes_to_copy_d2h;
          unsigned long long copy_offset_h2d;
          unsigned long long copy_offset_d2h;
          unsigned long long process_offset;

          // Benchmark approach 2: WITH streaming, looping over different
          // operations
          cudaEvent_t start_with_streaming, end_with_streaming;
          float elapsed_time_with_streaming;
          cudaEventCreate(&start_with_streaming);
          cudaEventCreate(&end_with_streaming);
          cudaEventRecord(start_with_streaming);

          for (int i = 0; i < nStreams; ++i) {
            process_offset = i * streamInfo.stream_size;
            copy_offset_d2h = i * streamInfo.stream_size;
            bytes_to_copy_d2h = stream_bytes;

            if(nStreams > 1) {
              if (i == 0) {
                copy_offset_h2d = 0;
                bytes_to_copy_h2d = stream_bytes + vol_slice_bytes;
              }
              else if (1 <= i && i < nStreams - 1) {
                copy_offset_h2d = i* streamInfo.stream_size - N*M;
                bytes_to_copy_h2d = stream_bytes + 2*vol_slice_bytes;
              }
              else {
                copy_offset_h2d = i* streamInfo.stream_size - N*M;
                bytes_to_copy_h2d = leftover_slab_bytes + vol_slice_bytes;
                bytes_to_copy_d2h = leftover_slices * N*M * sizeof(float);
              }
            }
            else {
              copy_offset_h2d = 0;
              bytes_to_copy_h2d = stream_bytes;
            }

            // copy a slab of the data from CPU to GPU memory
            checkCudaErrors( cudaMemcpyAsync(&d_input0[copy_offset_h2d], &pinned_mem_input[copy_offset_h2d],
                                      bytes_to_copy_h2d, cudaMemcpyHostToDevice,
                                      stream[i]) );

            /* Full data (traditional) 3D case */
            if (kernel_size == 3) medfilt1_kernel_3D<<<dimGrid,dimBlock,0,stream[i]>>>(d_input0, d_output0, process_offset, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
            else if (kernel_size == 5) medfilt2_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
            else if (kernel_size == 7) medfilt3_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
            else if (kernel_size == 9) medfilt4_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
            else medfilt5_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);

            // copy a slab of the processed data from GPU to CPU memory
            checkCudaErrors( cudaMemcpyAsync(&pinned_mem_output[copy_offset_d2h], &d_output0[copy_offset_d2h],
                                      bytes_to_copy_d2h, cudaMemcpyDeviceToHost,
                                      stream[i]) );
          }
          // finish clocking approach 2: WITH streaming, looping over different
          // operations
          cudaEventRecord(end_with_streaming);
          cudaEventSynchronize(end_with_streaming);
          cudaEventElapsedTime(&elapsed_time_with_streaming,
            start_with_streaming, end_with_streaming);
          printf("3D median filter WITH %d streams and looping over different operations took %f ms\n", nStreams,
            elapsed_time_with_streaming);
          cudaEventDestroy(start_with_streaming);
          cudaEventDestroy(end_with_streaming);


          // Benchmark approach 3: WITH streaming, looping over like-operations
          cudaEvent_t start_with_streaming_like_ops, end_with_streaming_like_ops;
          float elapsed_time_with_streaming_like_ops;
          cudaEventCreate(&start_with_streaming_like_ops);
          cudaEventCreate(&end_with_streaming_like_ops);
          cudaEventRecord(start_with_streaming_like_ops);

          // loop over h2d copying
          for (int i = 0; i < nStreams; i++) {
            copy_offset_d2h = i * streamInfo.stream_size;
            bytes_to_copy_d2h = stream_bytes;

            if(nStreams > 1) {
              if (i == 0) {
                copy_offset_h2d = 0;
                bytes_to_copy_h2d = stream_bytes + vol_slice_bytes;
              }
              else if (1 <= i && i < nStreams - 1) {
                copy_offset_h2d = i* streamInfo.stream_size - N*M;
                bytes_to_copy_h2d = stream_bytes + 2*vol_slice_bytes;
              }
              else {
                copy_offset_h2d = i* streamInfo.stream_size - N*M;
                bytes_to_copy_h2d = leftover_slab_bytes + vol_slice_bytes;
                bytes_to_copy_d2h = leftover_slices * N*M * sizeof(float);
              }
            }
            else {
              copy_offset_h2d = 0;
              bytes_to_copy_h2d = stream_bytes;
            }

            // copy a slab of the data from CPU to GPU memory
            checkCudaErrors( cudaMemcpyAsync(&d_input0[copy_offset_h2d], &pinned_mem_input[copy_offset_h2d],
                                      bytes_to_copy_h2d, cudaMemcpyHostToDevice,
                                      stream[i]) );
          }

          // loop over kernel launches
          for (int i = 0; i < nStreams; i++) {
            process_offset = i * streamInfo.stream_size;
            medfilt1_kernel_3D<<<dimGrid,dimBlock,0,stream[i]>>>(d_input0, d_output0, process_offset, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
          }

          // loop over d2h copying
          for (int i = 0; i < nStreams; i++) {
            copy_offset_d2h = i * streamInfo.stream_size;
            bytes_to_copy_d2h = stream_bytes;

            if(nStreams > 1) {
              if (i == 0) {
                copy_offset_h2d = 0;
                bytes_to_copy_h2d = stream_bytes + vol_slice_bytes;
              }
              else if (1 <= i && i < nStreams - 1) {
                copy_offset_h2d = i* streamInfo.stream_size - N*M;
                bytes_to_copy_h2d = stream_bytes + 2*vol_slice_bytes;
              }
              else {
                copy_offset_h2d = i* streamInfo.stream_size - N*M;
                bytes_to_copy_h2d = leftover_slab_bytes + vol_slice_bytes;
                bytes_to_copy_d2h = leftover_slices * N*M * sizeof(float);
              }
            }
            else {
              copy_offset_h2d = 0;
              bytes_to_copy_h2d = stream_bytes;
            }

            // copy a slab of the processed data from GPU to CPU memory
            checkCudaErrors( cudaMemcpyAsync(&pinned_mem_output[copy_offset_d2h], &d_output0[copy_offset_d2h],
                                      bytes_to_copy_d2h, cudaMemcpyDeviceToHost,
                                      stream[i]) );
          }

          // finish clocking approach 3: WITH streaming, looping over
          // like-operations
          cudaEventRecord(end_with_streaming_like_ops);
          cudaEventSynchronize(end_with_streaming_like_ops);
          cudaEventElapsedTime(&elapsed_time_with_streaming_like_ops,
            start_with_streaming_like_ops, end_with_streaming_like_ops);
          printf("3D median filter WITH %d streams and looping over like-operatons took %f ms\n",
            nStreams, elapsed_time_with_streaming_like_ops);
          cudaEventDestroy(start_with_streaming_like_ops);
          cudaEventDestroy(end_with_streaming_like_ops);

        }
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() );
    		}

        // when using pinned memory to hold output, copy result from pinned
        // memeory to paged memory (the Output pointer) so then the result is
        // seen by the python wrapper
        for (unsigned long long i = 0; i < ImSize; i++) {
          Output[i] = pinned_mem_output[i];
        }

        /*destroy streams*/
        for (int i = 0; i < nStreams; ++i)
          checkCudaErrors( cudaStreamDestroy(stream[i]) );

        /*free GPU memory*/
        checkCudaErrors(cudaFree(d_input0));
        checkCudaErrors(cudaFree(d_output0));

        // free pinned memory on host
        cudaFreeHost(pinned_mem_input);
        cudaFreeHost(pinned_mem_output);

        /*CHECK(cudaMemcpy(Output,d_output0,ImSize*sizeof(float),cudaMemcpyDeviceToHost));*/
        //cudaDeviceReset();
        return 0;
}

extern "C" int MedianFilt_GPU_main_uint16(unsigned short *Input, unsigned short *Output, int kernel_size, float mu_threshold, int N, int M, int Z)
{
  int deviceCount = -1; // number of devices
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
      fprintf(stderr, "No CUDA devices found\n");
       return -1;
   }
        int ImSize, sizefilter_total, kernel_half_size, midval;
        unsigned short *d_input, *d_output;
        ImSize = N*M*Z;

        checkCudaErrors(cudaMalloc((void**)&d_input,ImSize*sizeof(unsigned short)));
        checkCudaErrors(cudaMalloc((void**)&d_output,ImSize*sizeof(unsigned short)));

        checkCudaErrors(cudaMemcpy(d_input,Input,ImSize*sizeof(unsigned short),cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_output,Input,ImSize*sizeof(unsigned short),cudaMemcpyHostToDevice));

	if (Z == 1) {
        /*2D case */
        dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
        dim3 dimGrid(idivup(N,BLKXSIZE2D), idivup(M,BLKYSIZE2D));
        sizefilter_total = (int)(pow(kernel_size,2));
        kernel_half_size = (int)((kernel_size-1)/2);
        midval = (int)(sizefilter_total/2);

        if (kernel_size == 3) medfilt1_kernel_uint16_2D<<<dimGrid,dimBlock>>>(d_input, d_output, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
        else if (kernel_size == 5) medfilt2_kernel_uint16_2D<<<dimGrid,dimBlock>>>(d_input, d_output, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
        else if (kernel_size == 7) medfilt3_kernel_uint16_2D<<<dimGrid,dimBlock>>>(d_input, d_output, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
        else if (kernel_size == 9) medfilt4_kernel_uint16_2D<<<dimGrid,dimBlock>>>(d_input, d_output, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
        else medfilt5_kernel_uint16_2D<<<dimGrid,dimBlock>>>(d_input, d_output, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() );
       }
	else {
		    /*3D case*/
        dim3 dimBlock(BLKXSIZE,BLKYSIZE,BLKZSIZE);
        dim3 dimGrid(idivup(N,BLKXSIZE), idivup(M,BLKYSIZE),idivup(Z,BLKXSIZE));
        sizefilter_total = (int)(pow(kernel_size, 3));
        kernel_half_size = (int)((kernel_size-1)/2);
        midval = (int)(sizefilter_total/2);

        if (Z == kernel_size) {
        /* performs operation only on the central frame using all 3D information */
        if (kernel_size == 3) medfilt1_pad_kernel_uint16_3D<<<dimGrid,dimBlock>>>(d_input, d_output, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        else if (kernel_size == 5) medfilt2_pad_kernel_uint16_3D<<<dimGrid,dimBlock>>>(d_input, d_output, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        else if (kernel_size == 7) medfilt3_pad_kernel_uint16_3D<<<dimGrid,dimBlock>>>(d_input, d_output, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        else if (kernel_size == 9) medfilt4_pad_kernel_uint16_3D<<<dimGrid,dimBlock>>>(d_input, d_output, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        else medfilt5_pad_kernel_uint16_3D<<<dimGrid,dimBlock>>>(d_input, d_output, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
          }
        else {
        /* Full data (traditional) 3D case */
        if (kernel_size == 3) medfilt1_kernel_uint16_3D<<<dimGrid,dimBlock>>>(d_input, d_output, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        else if (kernel_size == 5) medfilt2_kernel_uint16_3D<<<dimGrid,dimBlock>>>(d_input, d_output, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        else if (kernel_size == 7) medfilt3_kernel_uint16_3D<<<dimGrid,dimBlock>>>(d_input, d_output, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        else if (kernel_size == 9) medfilt4_kernel_uint16_3D<<<dimGrid,dimBlock>>>(d_input, d_output, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        else medfilt5_kernel_uint16_3D<<<dimGrid,dimBlock>>>(d_input, d_output, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
            }
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() );
    		}
        checkCudaErrors(cudaMemcpy(Output,d_output,ImSize*sizeof(unsigned short),cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_input));
        checkCudaErrors(cudaFree(d_output));
        return 0;
}
