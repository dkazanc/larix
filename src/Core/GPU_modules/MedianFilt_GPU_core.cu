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
 * 3. mu_threshold: if not a zero value then deinzger
 
 * Output:
 * [1] Filtered or dezingered image/volume
 *
 */

#define BLKXSIZE 8
#define BLKYSIZE 8
#define BLKZSIZE 8

#define BLKXSIZE2D 16
#define BLKYSIZE2D 16
#define EPS 1.0e-5

#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
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
        int ImSize;
        float *d_input, *d_output;
        ImSize = N*M*Z;

        CHECK(cudaMalloc((void**)&d_input,ImSize*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_output,ImSize*sizeof(float)));

        CHECK(cudaMemcpy(d_input,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
        /*CHECK(cudaMemcpy(d_output,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));*/

	if (Z == 1) {
        /*2D case */
        dim3 dimBlock(BLKXSIZE2D,BLKYSIZE2D);
        dim3 dimGrid(idivup(N,BLKXSIZE2D), idivup(M,BLKYSIZE2D));

        copy_float_array_kernel_2D<<<dimGrid,dimBlock>>>(d_input, d_output, N, M, ImSize);
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
