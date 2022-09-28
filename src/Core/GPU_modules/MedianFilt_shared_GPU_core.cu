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

#include "MedianFilt_shared_GPU_core.h"
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

__global__ void global3d_kernel(float *Input, float* Output, int kernel_half_size, int sizefilter_total, int midval, int N, int M, int Z, int num_total)
    {
      float ValVec[CONSTVECSIZE_9];
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
      //quicksort_float(ValVec, 0, 8); /* perform sorting */
      sort_bubble(ValVec, CONSTVECSIZE_9); /* perform sorting */

        Output[index] = ValVec[midval]; /* perform median filtration */
      }
      return;
    }

// this function is used to shift elements along the Z local column
inline __device__ void advance(float *field, const int num_points)
{
#pragma unroll
  for(int i=0; i<num_points; i++)
    field[i] = field[i+1];
}


    // The kernel will work on an input which has been already padded with radius elements
    // so the "in" parameter points to the first element to be process inside an inner volume
    // of side nx,ny,nz with stride stride_x, which in general coulb be stride_x = nx + 2*radius

template <int radius, int diameter> // diameter = 2*radius+1
inline __device__ void medfilt_kernel_3D_t(
    float *in,
    float *out,
    int nx, // faster dimension
    int ny,
    int nz, // slower dimension
    int stride_x, // stride along faster dimension
    int stride_yx // stride along each plane
    )
{
  __shared__ float s_data[TILEDIMY+2*radius][TILEDIMX+2*radius];
  float ValVec[diameter*diameter*diameter]; // I'd suggest to put this in global and reuse it plane by plane
  float *input  = (float*) in;
  float *output = (float*) out;
  float *input_prefill  = (float*) in;

  int tx = threadIdx.x + radius;
  int ty = threadIdx.y + radius;
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  input  += iy*stride_x + ix;
  output += iy*stride_x + ix;
  bool im_in_x = ix < nx;
  bool im_in_y = iy < ny;
  int offset = -radius*stride_yx;
  input_prefill = input + offset; // hypotesys: in volume surrounded by padding
  float local_input[diameter];

  // prefill Z columns per thread skipping the first element since 
  // it will be filled using the advance function in main loop along Z
  for(int i=1; i<diameter; i++)
  {
    local_input[i] = *input_prefill;
    input_prefill += stride_yx;
  }
  for(int iz=0; iz<nz; iz++)
  {
    // shift elements and insert a new one on top
    advance( local_input, diameter-1 );
    local_input[diameter-1] = *input_prefill;
    int index = 0;
    // loading local neighbours of thread (tx,ty) plane by plane
    for(int rz = -radius; rz < radius; rz++) {
      // make sure all threads have completed previous plane
      __syncthreads();
      // filling the bulk elements of the tile
      if (im_in_x && im_in_y)     s_data[ty][tx] = local_input[rz];
      // fill halos
      // TODO: for the moment halos are read directly from global memory, 
      //       we should find a better trick to lower global memory accesses
      // along x lower
      if (tx < radius) {
        s_data[ty][tx-radius] = input[rz*stride_yx-radius];
      }
      // along x upper
      if (tx > TILEDIMX - radius && im_in_x) {
        s_data[ty][tx+radius] = input[rz*stride_yx+radius];
      }
      // along y upper
      if (ty < radius) {
        s_data[ty-radius][tx] = input[rz*stride_yx-radius*stride_x];
      }
      // along y lower
      if (ty > TILEDIMY - radius && im_in_y) {
        s_data[ty+radius][tx] = input[rz*stride_yx+radius*stride_x];
      }
      // filling corners
      if (tx < radius && ty < radius) {
        s_data[ty-radius][tx-radius] = input[rz*stride_yx-radius*stride_x-radius];
      }
      if (tx > TILEDIMX - radius && ty < radius && im_in_x) {
        s_data[ty-radius][tx+radius] = input[rz*stride_yx-radius*stride_x+radius];
      }
      if (tx < radius && ty > TILEDIMY - radius && im_in_y) {
        s_data[ty+radius][tx-radius] = input[rz*stride_yx+radius*stride_x-radius];
      }
      if (tx > TILEDIMX - radius && ty > TILEDIMY - radius && im_in_x && im_in_y) {
        s_data[ty+radius][tx+radius] = input[rz*stride_yx+radius*stride_x+radius];
      }
      // make sure all elements are in shared memory
      __syncthreads();
      for(int ry = -radius; ry < radius; ry++) {
        for(int rx = -radius; rx < radius; rx++) {
          ValVec[index++] = s_data[ty+ry][tx+rx];
        }
      }
    } // move to next local plane
         
    if (im_in_x && im_in_y) *output = ValVec[radius+radius*diameter+radius*diameter*diameter];
    // update pointer to next plane
    input  += stride_yx;
    output += stride_yx;
    input_prefill  += stride_yx;
    }
} 
////////////////////////////////////////////////
/////////////// HOST FUNCTION ///////////////////
/////////////////////////////////////////////////
extern "C"
__global__ void medfilt_kernel_3D_r5(
    float *input,
    float *output,
    int nx, int ny, int nz,
    int stride_x, int stride_yx
    )
{
  /*switching here different radii*/
  const int diameter = 2*1+1;
  medfilt_kernel_3D_t<1, diameter>(
      input, output,
      nx, ny, nz,
      stride_x, stride_yx
      );
}

extern "C" int MedianFilt_shared_GPU_main_float32(float *Input, float *Output, int kernel_size, int gpu_device, int N, int M, int Z)
{
  int deviceCount = -1; // number of devices
  cudaGetDeviceCount(&deviceCount);
  /*set GPU device*/
  checkCudaErrors(cudaSetDevice(gpu_device));  

  if (deviceCount == 0) {
      fprintf(stderr, "No CUDA devices found\n");
       return -1;
   }
        int ImSize, sizefilter_total, kernel_half_size, midval;
        float *d_input, *d_output;
        ImSize = N*M*Z;

        checkCudaErrors(cudaMalloc((void**)&d_input,ImSize*sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&d_output,ImSize*sizeof(float)));

        checkCudaErrors(cudaMemcpy(d_input,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_output,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));

	    /*3D case*/
        dim3 dimBlock(BLKXSIZE,BLKYSIZE,BLKZSIZE);
        dim3 dimGrid(idivup(N,BLKXSIZE), idivup(M,BLKYSIZE),idivup(Z,BLKXSIZE));
        sizefilter_total = (int)(pow(kernel_size, 3));
        kernel_half_size = (int)((kernel_size-1)/2);
        midval = (int)(sizefilter_total/2);

        /* Full data (traditional) 3D case */
        //if (kernel_size == 3) global3d_kernel<<<dimGrid,dimBlock>>>(d_input, d_output, kernel_half_size, sizefilter_total, midval, N, M, Z, ImSize);
        printf("%i %i %i\n", N, M, Z);
        dim3 threads = dim3(TILEDIMX,TILEDIMY);
        dim3 blocks(N/threads.x+1, M/threads.y+1);
        const int stride_x = N;
        const int stride_yx = N*M;
        medfilt_kernel_3D_r5<<<threads,blocks>>>(d_input, d_output, N-2*kernel_half_size, M-2*kernel_half_size, Z-2*kernel_half_size, stride_x, stride_yx);

        checkCudaErrors( cudaDeviceSynchronize() );
        checkCudaErrors(cudaPeekAtLastError() );

        checkCudaErrors(cudaMemcpy(Output,d_output,ImSize*sizeof(float),cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_input));
        checkCudaErrors(cudaFree(d_output));
        return 0;    
}