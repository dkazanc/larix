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
/* CUDA implementation of the median filtration and dezingering (2D/3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume [REQUIRED]
 * 2. filter_half_window_size: The half size of the median filter window
 * 3. mu_threshold: if not a zero value then dezing

 * Output:
 * [1] Filtered or dezingered image/volume
 */
/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
__global__ void medfilt1_kernel_2D(float *Input, float* Output, int filter_half_window_size, int sizefilter_total, float mu_threshold, int N, int M, int num_total)
  {
      float ValVec[CONSTVECSIZE_9];
      int i1, j1, i_m, j_m, midval;
      midval = (int)(sizefilter_total*0.5f) - 1;

      const int i = blockDim.x * blockIdx.x + threadIdx.x;
      const int j = blockDim.y * blockIdx.y + threadIdx.y;
      const int index = i + N*j;

      if (index < num_total)	{
        if ((i >= filter_half_window_size)  && (i < N-filter_half_window_size))	{
          if ((j >= filter_half_window_size)  && (j < M-filter_half_window_size))	{
      int counter = 0;
      for(i_m=-filter_half_window_size; i_m<=filter_half_window_size; i_m++) {
            i1 = i + i_m;
            for(j_m=-filter_half_window_size; j_m<=filter_half_window_size; j_m++) {
              j1 = j + j_m;
              ValVec[counter++] = Input[i1 + N*j1];
      }}
      //sort_quick(ValVec, 0, CONSTVECSIZE_9); /* perform sorting */
      sort_bubble(ValVec, CONSTVECSIZE_9); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
      }
      }}
      }
  }

__global__ void medfilt2_kernel_2D(float *Input, float* Output, int filter_half_window_size, int sizefilter_total, float mu_threshold, int N, int M, int num_total)
    {
        float ValVec[CONSTVECSIZE_25];
        int i1, j1, i_m, j_m, midval;
        midval = (int)(sizefilter_total*0.5f) - 1;

        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        const int index = i + N*j;

        if (index < num_total)	{
          if ((i >= filter_half_window_size)  && (i < N-filter_half_window_size))	{
            if ((j >= filter_half_window_size)  && (j < M-filter_half_window_size))	{
        int counter = 0;
        for(i_m=-filter_half_window_size; i_m<=filter_half_window_size; i_m++) {
              i1 = i + i_m;
              for(j_m=-filter_half_window_size; j_m<=filter_half_window_size; j_m++) {
                j1 = j + j_m;
                ValVec[counter++] = Input[i1 + N*j1];
        }}
        //sort_quick(ValVec, 0, CONSTVECSIZE_25); /* perform sorting */
        sort_bubble(ValVec, CONSTVECSIZE_25); /* perform sorting */
        if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
        else {
        /* perform dezingering */
        if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
        }
          }}
        }
    }

__global__ void medfilt3_kernel_2D(float *Input, float* Output, int filter_half_window_size, int sizefilter_total, float mu_threshold, int N, int M, int num_total)
    {
        float ValVec[CONSTVECSIZE_49];
        int i1, j1, i_m, j_m, midval;
        midval = (int)(sizefilter_total*0.5f) - 1;

        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        const int index = i + N*j;

        if (index < num_total)	{
          if ((i >= filter_half_window_size)  && (i < N-filter_half_window_size))	{
            if ((j >= filter_half_window_size)  && (j < M-filter_half_window_size))	{
        int counter = 0;
        for(i_m=-filter_half_window_size; i_m<=filter_half_window_size; i_m++) {
              i1 = i + i_m;
              for(j_m=-filter_half_window_size; j_m<=filter_half_window_size; j_m++) {
                j1 = j + j_m;
                ValVec[counter++] = Input[i1 + N*j1];
        }}
        //sort_quick(ValVec, 0, CONSTVECSIZE_49); /* perform sorting */
        sort_bubble(ValVec, CONSTVECSIZE_49); /* perform sorting */
        if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
        else {
        /* perform dezingering */
        if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
        }
          }}
        }
    }

__global__ void medfilt4_kernel_2D(float *Input, float* Output, int filter_half_window_size, int sizefilter_total, float mu_threshold, int N, int M, int num_total)
    {
        float ValVec[CONSTVECSIZE_81];
        int i1, j1, i_m, j_m, midval;
        midval = (int)(sizefilter_total*0.5f) - 1;

        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        const int index = i + N*j;

        if (index < num_total)	{
          if ((i >= filter_half_window_size)  && (i < N-filter_half_window_size))	{
            if ((j >= filter_half_window_size)  && (j < M-filter_half_window_size))	{
        int counter = 0;
        for(i_m=-filter_half_window_size; i_m<=filter_half_window_size; i_m++) {
              i1 = i + i_m;
              for(j_m=-filter_half_window_size; j_m<=filter_half_window_size; j_m++) {
                j1 = j + j_m;
                ValVec[counter++] = Input[i1 + N*j1];
        }}
        //sort_quick(ValVec, 0, CONSTVECSIZE_81); /* perform sorting */
        sort_bubble(ValVec, CONSTVECSIZE_81); /* perform sorting */
        if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
        else {
        /* perform dezingering */
        if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
        }
          }}
        }
    }

__global__ void medfilt5_kernel_2D(float *Input, float* Output, int filter_half_window_size, int sizefilter_total, float mu_threshold, int N, int M, int num_total)
    {
        float ValVec[CONSTVECSIZE_121];
        int i1, j1, i_m, j_m, midval;
        midval = (int)(sizefilter_total*0.5f) - 1;

        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        const int index = i + N*j;

        if (index < num_total)	{
          if ((i >= filter_half_window_size)  && (i < N-filter_half_window_size))	{
            if ((j >= filter_half_window_size)  && (j < M-filter_half_window_size))	{
        int counter = 0;
        for(i_m=-filter_half_window_size; i_m<=filter_half_window_size; i_m++) {
              i1 = i + i_m;
              for(j_m=-filter_half_window_size; j_m<=filter_half_window_size; j_m++) {
                j1 = j + j_m;
                ValVec[counter++] = Input[i1 + N*j1];
        }}
        //sort_quick(ValVec, 0, CONSTVECSIZE_121); /* perform sorting */
        sort_bubble(ValVec, CONSTVECSIZE_121); /* perform sorting */
        if (mu_threshold == 0.0f) Output[index] = ValVec[midval]; /* perform median filtration */
        else {
        /* perform dezingering */
        if (abs(Input[index] - ValVec[midval]) >= mu_threshold) Output[index] = ValVec[midval];
        }
          }}
        }
    }

__global__ void copy_float_array_kernel_2D(float *Input, float* Output, int N, int M, int num_total)
  {
      int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
      int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

      int index = xIndex + N*yIndex;

      if (index < num_total)	{
          Output[index] = Input[index];
      }
  }


/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/

/////////////////////////////////////////////////
// HOST FUNCTION
extern "C" int MedianFilt_GPU_main(float *Input, float *Output, int filter_half_window_size, float mu_threshold, int N, int M, int Z)
{
  int deviceCount = -1; // number of devices
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
      fprintf(stderr, "No CUDA devices found\n");
       return -1;
   }
        int ImSize, sizefilter_total;
        float *d_input, *d_output;
        ImSize = N*M*Z;

        CHECK(cudaMalloc((void**)&d_input,ImSize*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_output,ImSize*sizeof(float)));

        CHECK(cudaMemcpy(d_input,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_output,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));

	if (Z == 1) {
        /*2D case */
        dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
        dim3 dimGrid(idivup(N,BLKXSIZE2D), idivup(M,BLKYSIZE2D));
        sizefilter_total = (2*filter_half_window_size + 1)*(2*filter_half_window_size + 1);

        if (filter_half_window_size == 1) medfilt1_kernel_2D<<<dimGrid,dimBlock>>>(d_input, d_output, filter_half_window_size, sizefilter_total, mu_threshold, N, M, ImSize);
        else if (filter_half_window_size == 2) medfilt2_kernel_2D<<<dimGrid,dimBlock>>>(d_input, d_output, filter_half_window_size, sizefilter_total, mu_threshold, N, M, ImSize);
        else if (filter_half_window_size == 3) medfilt3_kernel_2D<<<dimGrid,dimBlock>>>(d_input, d_output, filter_half_window_size, sizefilter_total, mu_threshold, N, M, ImSize);
        else if (filter_half_window_size == 4) medfilt4_kernel_2D<<<dimGrid,dimBlock>>>(d_input, d_output, filter_half_window_size, sizefilter_total, mu_threshold, N, M, ImSize);
        else medfilt5_kernel_2D<<<dimGrid,dimBlock>>>(d_input, d_output, filter_half_window_size, sizefilter_total, mu_threshold, N, M, ImSize);
        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() );        
       }
	else {
		/*3D case*/
		}
        CHECK(cudaMemcpy(Output,d_output,ImSize*sizeof(float),cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_output));

        return 0;
}
