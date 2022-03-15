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
__global__ void medfilt1_kernel_2D(float *Input, float* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int num_total)
  {
      float ValVec[CONSTVECSIZE_9];
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
      sort_bubble(ValVec, CONSTVECSIZE_9); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
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
__global__ void medfilt1_kernel_3D(float *Input, float* Output, int kernel_half_size, int sizefilter_total, float mu_threshold, int midval, int N, int M, int Z, int num_total)
  {
      float ValVec[CONSTVECSIZE_27];
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

        int ImSize, sizefilter_total, kernel_half_size, midval;
        ImSize = N*M*Z;

        /* adapted to work with 8 GPU devices in total */
        float *d_input0, *d_output0, *d_input1, *d_output1;

        cudaSetDevice(gpu_device);
        if (gpu_device == 0) {
        CHECK(cudaMalloc((void**)&d_input0,ImSize*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_output0,ImSize*sizeof(float)));
        CHECK(cudaMemcpy(d_input0,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_output0,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
        }
        if (gpu_device == 1) {
        CHECK(cudaMalloc((void**)&d_input1,ImSize*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_output1,ImSize*sizeof(float)));
        CHECK(cudaMemcpy(d_input1,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_output1,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
        }

	if (Z == 1) {
        /*2D case */
        dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
        dim3 dimGrid(idivup(N,BLKXSIZE2D), idivup(M,BLKYSIZE2D));
        sizefilter_total = (int)(pow(kernel_size,2));
        kernel_half_size = (int)((kernel_size-1)/2);
        midval = (int)(sizefilter_total/2);

        if (kernel_size == 3) {
            if (gpu_device == 0) medfilt1_kernel_2D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
            if (gpu_device == 1) medfilt1_kernel_2D<<<dimGrid,dimBlock>>>(d_input1, d_output1, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
        }
        else if (kernel_size == 5) {
            if (gpu_device == 0) medfilt2_kernel_2D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
            if (gpu_device == 1) medfilt2_kernel_2D<<<dimGrid,dimBlock>>>(d_input1, d_output1, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
        }
        else if (kernel_size == 7) {
            if (gpu_device == 0) medfilt3_kernel_2D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
            if (gpu_device == 1) medfilt3_kernel_2D<<<dimGrid,dimBlock>>>(d_input1, d_output1, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
        }
        else if (kernel_size == 9) {
            if (gpu_device == 0) medfilt4_kernel_2D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
            if (gpu_device == 1) medfilt4_kernel_2D<<<dimGrid,dimBlock>>>(d_input1, d_output1, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
        }
        else {
          if (gpu_device == 0) medfilt5_kernel_2D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
          if (gpu_device == 1) medfilt5_kernel_2D<<<dimGrid,dimBlock>>>(d_input1, d_output1, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, ImSize);
        }
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
        if (kernel_size == 3) {
          if (gpu_device == 0) medfilt1_pad_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
          if (gpu_device == 1) medfilt1_pad_kernel_3D<<<dimGrid,dimBlock>>>(d_input1, d_output1, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        }
        else if (kernel_size == 5) {
          if (gpu_device == 0) medfilt2_pad_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
          if (gpu_device == 1) medfilt2_pad_kernel_3D<<<dimGrid,dimBlock>>>(d_input1, d_output1, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        }
        else if (kernel_size == 7) {
          if (gpu_device == 0) medfilt3_pad_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
          if (gpu_device == 1) medfilt3_pad_kernel_3D<<<dimGrid,dimBlock>>>(d_input1, d_output1, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        }
        else if (kernel_size == 9) {
           if (gpu_device == 0) medfilt4_pad_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
           if (gpu_device == 1) medfilt4_pad_kernel_3D<<<dimGrid,dimBlock>>>(d_input1, d_output1, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
         }
        else {
          if (gpu_device == 0) medfilt5_pad_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
          if (gpu_device == 1) medfilt5_pad_kernel_3D<<<dimGrid,dimBlock>>>(d_input1, d_output1, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        }
          }
        else {
        /* Full data (traditional) 3D case */
        if (kernel_size == 3) {
          if (gpu_device == 0) medfilt1_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
          if (gpu_device == 1) medfilt1_kernel_3D<<<dimGrid,dimBlock>>>(d_input1, d_output1, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        }
        else if (kernel_size == 5) {
          if (gpu_device == 0) medfilt2_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
          if (gpu_device == 1) medfilt2_kernel_3D<<<dimGrid,dimBlock>>>(d_input1, d_output1, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        }
        else if (kernel_size == 7) {
          if (gpu_device == 0) medfilt3_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
          if (gpu_device == 1) medfilt3_kernel_3D<<<dimGrid,dimBlock>>>(d_input1, d_output1, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        }
        else if (kernel_size == 9) {
           if (gpu_device == 0) medfilt4_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
           if (gpu_device == 1) medfilt4_kernel_3D<<<dimGrid,dimBlock>>>(d_input1, d_output1, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
         }
        else {
          if (gpu_device == 0) medfilt5_kernel_3D<<<dimGrid,dimBlock>>>(d_input0, d_output0, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
          if (gpu_device == 1) medfilt5_kernel_3D<<<dimGrid,dimBlock>>>(d_input1, d_output1, kernel_half_size, sizefilter_total, mu_threshold, midval, N, M, Z, ImSize);
        }
            }
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() );
    		}

        if (gpu_device == 0) {
        CHECK(cudaMemcpy(Output,d_output0,ImSize*sizeof(float),cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_input0));
        CHECK(cudaFree(d_output0));
        }
        if (gpu_device == 1) {
        CHECK(cudaMemcpy(Output,d_output1,ImSize*sizeof(float),cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_input1));
        CHECK(cudaFree(d_output1));
        }
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

        CHECK(cudaMalloc((void**)&d_input,ImSize*sizeof(unsigned short)));
        CHECK(cudaMalloc((void**)&d_output,ImSize*sizeof(unsigned short)));

        CHECK(cudaMemcpy(d_input,Input,ImSize*sizeof(unsigned short),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_output,Input,ImSize*sizeof(unsigned short),cudaMemcpyHostToDevice));

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
        CHECK(cudaMemcpy(Output,d_output,ImSize*sizeof(unsigned short),cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_output));
        return 0;
}
