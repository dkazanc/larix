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

  template <int radius, int diameter> // diameter should be set to 2*radius+1
inline __device__ void medfilt_kernel_2D_t(
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
  float ValVec[diameter*diameter]; // 2D

  float *input  = (float*) in;
  float *output = (float*) out;

  int tx = threadIdx.x + radius;
  int ty = threadIdx.y + radius;
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  bool im_in_x = ix < nx;
  bool im_in_y = iy < ny;

  int tiledimx = TILEDIMX;
  int tiledimy = TILEDIMY;
  // resize tile if current threadblock is on the border
  if ( (blockIdx.x+1) * blockDim.x > nx) {
    tiledimx = nx % blockDim.x;
  }
  if ((blockIdx.y+1) * blockDim.y > ny) {
    tiledimy = ny % blockDim.y;
  }

  input  += iy*stride_x + ix;
  output += iy*stride_x + ix;

  for(int iz=0; iz<nz; iz++)
  {
    int index = 0;
    // loading local neighbours of thread (tx,ty) plane by plane
    {
      // make sure all threads have completed previous plane
      __syncthreads();

      {
        // filling the bulk elements of the tile
        if (im_in_x && im_in_y) { // threads inside inner volume
          s_data[ty][tx] = *input;
        }

        // fill halos
        // TODO: for the moment halos are read directly from global memory, 
        //       we should find a better trick to lower global memory accesses

        if (im_in_y) {
          if (threadIdx.x < radius ) {
            // along x lower
            s_data[ty][tx-radius] = input[-radius];
            // along x upper
            s_data[ty][tx+tiledimx] = input[+tiledimx];
          }
        }

        if (im_in_x) {
          if (threadIdx.y < radius) {
            // along y upper
            s_data[ty-radius][tx] = input[-radius*stride_x];
            // along y lower
            s_data[ty+tiledimy][tx] = input[+tiledimy*stride_x];
          }
        }

        // filling corners
        if (threadIdx.x < radius && threadIdx.y < radius) {
          s_data[ty-radius][tx-radius] = input[-radius*stride_x-radius];
          s_data[ty-radius][tx+tiledimx] = input[-radius*stride_x+tiledimx];
          s_data[ty+tiledimy][tx-radius] = input[+tiledimy*stride_x-radius];
          s_data[ty+tiledimy][tx+tiledimx] = input[+tiledimy*stride_x+tiledimx];
        }

      }

      // make sure all elements are in shared memory
      __syncthreads();

      for(int ry = -radius; ry <= radius; ry++) {
        for(int rx = -radius; rx <= radius; rx++) {
          ValVec[index++] = s_data[ty+ry][tx+rx];
        }
      }

#ifdef WITH_DEBUG
      if (threadIdx.x == nx/2 && threadIdx.y == ny/2) {
        printf("\n");
        for(int ty = 0; ty < TILEDIMY+2*radius; ty++) {
          printf("%2d] ", ty);
          for(int tx = 0; tx < TILEDIMX+2*radius; tx++) {
            printf("%4.0f ", s_data[ty][tx]);
          }
          printf("\n");
        }
        printf("ValVec centered at (%d, %d):\n", threadIdx.x, threadIdx.y);
        for (int idx = 0; idx<diameter*diameter*diameter; idx++) {
          printf("%2.0f ", ValVec[idx]);
        }
      }
#endif // WITH_DEBUG

    }

    const int midpoint = radius + radius*diameter;
    if (im_in_x && im_in_y) {
#ifndef WITH_DEBUG
      sort_bubble(ValVec, diameter*diameter*diameter);
#endif // WITH_DEBUG
      *output = ValVec[midpoint];
    }

    // update pointer to next plane
    input  += stride_yx;
    output += stride_yx;
  }
}
// instances of predefined kernel templates with selected radius/diameter
__global__ void medfilt_kernel_2D_r1(float *in, float *out, int nx, int ny, int nz, int stride_x, int stride_yx)
{
  medfilt_kernel_2D_t<1,3>(in, out, nx, ny, nz, stride_x, stride_yx);
}
__global__ void medfilt_kernel_2D_r2(float *in, float *out, int nx, int ny, int nz, int stride_x, int stride_yx)
{
  medfilt_kernel_2D_t<2,5>(in, out, nx, ny, nz, stride_x, stride_yx);
}
__global__ void medfilt_kernel_2D_r3(float *in, float *out, int nx, int ny, int nz, int stride_x, int stride_yx)
{
  medfilt_kernel_2D_t<3,7>(in, out, nx, ny, nz, stride_x, stride_yx);
}
__global__ void medfilt_kernel_2D_r4(float *in, float *out, int nx, int ny, int nz, int stride_x, int stride_yx)
{
  medfilt_kernel_2D_t<4,9>(in, out, nx, ny, nz, stride_x, stride_yx);
}
__global__ void medfilt_kernel_2D_r5(float *in, float *out, int nx, int ny, int nz, int stride_x, int stride_yx)
{
  medfilt_kernel_2D_t<5,11>(in, out, nx, ny, nz, stride_x, stride_yx);
}


extern "C"
void medfilt_kernel_2D(
    float *input,
    float *output,
    int nx, int ny, int nz,
    int stride_x, int stride_yx,
    int radius
    )
{
  dim3 threads = dim3(TILEDIMX, TILEDIMY);
  dim3 blocks = dim3(nx/threads.x + 1, ny/threads.y + 1);
  switch (radius) {
    case 1:
      medfilt_kernel_2D_r1<<<blocks, threads>>>(input, output, nx, ny, nz, stride_x, stride_yx);
      break;
    case 2:
      medfilt_kernel_2D_r2<<<blocks, threads>>>(input, output, nx, ny, nz, stride_x, stride_yx);
      break;
    case 3:
      medfilt_kernel_2D_r3<<<blocks, threads>>>(input, output, nx, ny, nz, stride_x, stride_yx);
      break;
    case 4:
      medfilt_kernel_2D_r4<<<blocks, threads>>>(input, output, nx, ny, nz, stride_x, stride_yx);
      break;
    case 5:
      medfilt_kernel_2D_r5<<<blocks, threads>>>(input, output, nx, ny, nz, stride_x, stride_yx);
      break;
    default:
      fprintf(stderr,"ERROR: medilter_kernel_2D not implemented for radius=%d\n", radius);
  }
}

  template <int radius, int diameter> // diameter should be set to 2*radius+1
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
  float local_input[diameter];
  float ValVec[diameter*diameter*diameter]; // 3D

  float *input  = (float*) in;
  float *output = (float*) out;
  float *input_prefill  = (float*) in;

  int tx = threadIdx.x + radius;
  int ty = threadIdx.y + radius;
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;

  bool im_in_x = ix < nx;
  bool im_in_y = iy < ny;

  int tiledimx = TILEDIMX;
  int tiledimy = TILEDIMY;
  // resize tile if current threadblock is on the border
  if ( (blockIdx.x+1) * blockDim.x > nx) {
    tiledimx = nx % blockDim.x;
  }
  if ((blockIdx.y+1) * blockDim.y > ny) {
    tiledimy = ny % blockDim.y;
  }

  input  += iy*stride_x + ix;
  output += iy*stride_x + ix;
  input_prefill += iy*stride_x + ix;

  input_prefill -= radius*stride_yx; // start filling from radius planes lower
  // prefill Z column local_input per thread skipping the first element since 
  // it will be filled using the advance function in main loop over Z planes
  if (im_in_x && im_in_y) {
    for(int i=1; i<diameter; i++)
    {
      local_input[i] = *input_prefill;
      input_prefill += stride_yx;
    }
  }

  for(int iz=0; iz<nz; iz++)
  {
    // shift elements and insert a new one on top
    if (im_in_x && im_in_y) { // threads inside inner volume
      advance( local_input, diameter-1 );
      local_input[diameter-1] = *input_prefill;
    }

    int index = 0;
    // loading local neighbours of thread (tx,ty) plane by plane
    for(int rz = -radius; rz <= radius; rz++) 
    {
      // make sure all threads have completed previous plane
      __syncthreads();

      {
        // filling the bulk elements of the tile
        if (im_in_x && im_in_y) { // threads inside inner volume
          s_data[ty][tx] = local_input[radius+rz];
        }

        // fill halos
        // TODO: for the moment halos are read directly from global memory, 
        //       we should find a better trick to lower global memory accesses

        if (im_in_y) {
          if (threadIdx.x < radius ) {
            // along x lower
            s_data[ty][tx-radius] = input[rz*stride_yx-radius];
            // along x upper
            s_data[ty][tx+tiledimx] = input[rz*stride_yx+tiledimx];
          }
        }

        if (im_in_x) {
          if (threadIdx.y < radius) {
            // along y upper
            s_data[ty-radius][tx] = input[rz*stride_yx-radius*stride_x];
            // along y lower
            s_data[ty+tiledimy][tx] = input[rz*stride_yx+tiledimy*stride_x];
          }
        }

        // filling corners
        if (threadIdx.x < radius && threadIdx.y < radius) {
          s_data[ty-radius][tx-radius] = input[rz*stride_yx-radius*stride_x-radius];
          s_data[ty-radius][tx+tiledimx] = input[rz*stride_yx-radius*stride_x+tiledimx];
          s_data[ty+tiledimy][tx-radius] = input[rz*stride_yx+tiledimy*stride_x-radius];
          s_data[ty+tiledimy][tx+tiledimx] = input[rz*stride_yx+tiledimy*stride_x+tiledimx];
        }
      }

      // make sure all elements are in shared memory
      __syncthreads();

      for(int ry = -radius; ry <= radius; ry++) {
        for(int rx = -radius; rx <= radius; rx++) {
          ValVec[index++] = s_data[ty+ry][tx+rx];
        }
      }

#ifdef WITH_DEBUG
      if (threadIdx.x == nx/2 && threadIdx.y == ny/2) {
        printf("\n");
        for(int ty = 0; ty < TILEDIMY+2*radius; ty++) {
          printf("%2d] ", ty);
          for(int tx = 0; tx < TILEDIMX+2*radius; tx++) {
            printf("%4.0f ", s_data[ty][tx]);
          }
          printf("\n");
        }
        printf("ValVec centered at (%d, %d):\n", threadIdx.x, threadIdx.y);
        for (int idx = 0; idx<diameter*diameter*diameter; idx++) {
          printf("%2.0f ", ValVec[idx]);
        }
      }
#endif // WITH_DEBUG

    } // move to next local plane

    const int midpoint = radius + radius*diameter + radius*diameter*diameter;
    if (im_in_x && im_in_y) {
#ifndef WITH_DEBUG
      sort_bubble(ValVec, diameter*diameter*diameter);      
#endif // WITH_DEBUG
      *output = ValVec[midpoint];
    }

    // update pointer to next plane
    input  += stride_yx;
    output += stride_yx;
    input_prefill += stride_yx;
  }
}

// instances of predefined kernel templates with selected radius/diameter
__global__ void medfilt_kernel_3D_r1(float *in, float *out, int nx, int ny, int nz, int stride_x, int stride_yx)
{
  medfilt_kernel_3D_t<1,3>(in, out, nx, ny, nz, stride_x, stride_yx);
}
__global__ void medfilt_kernel_3D_r2(float *in, float *out, int nx, int ny, int nz, int stride_x, int stride_yx)
{
  medfilt_kernel_3D_t<2,5>(in, out, nx, ny, nz, stride_x, stride_yx);
}
__global__ void medfilt_kernel_3D_r3(float *in, float *out, int nx, int ny, int nz, int stride_x, int stride_yx)
{
  medfilt_kernel_3D_t<3,7>(in, out, nx, ny, nz, stride_x, stride_yx);
}
__global__ void medfilt_kernel_3D_r4(float *in, float *out, int nx, int ny, int nz, int stride_x, int stride_yx)
{
  medfilt_kernel_3D_t<4,9>(in, out, nx, ny, nz, stride_x, stride_yx);
}
__global__ void medfilt_kernel_3D_r5(float *in, float *out, int nx, int ny, int nz, int stride_x, int stride_yx)
{
  medfilt_kernel_3D_t<5,11>(in, out, nx, ny, nz, stride_x, stride_yx);
}

extern "C"
void medfilt_kernel_3D(
    float *input,
    float *output,
    int nx, int ny, int nz,
    int stride_x, int stride_yx,
    int radius
    )
{
  dim3 threads = dim3(TILEDIMX, TILEDIMY);
  dim3 blocks = dim3(nx/threads.x + 1, ny/threads.y + 1);
  switch (radius) {
    case 1:
      medfilt_kernel_3D_r1<<<blocks, threads>>>(input, output, nx, ny, nz, stride_x, stride_yx);
      break;                                
    case 2:
      medfilt_kernel_3D_r2<<<blocks, threads>>>(input, output, nx, ny, nz, stride_x, stride_yx);
      break;
    case 3:
      medfilt_kernel_3D_r3<<<blocks, threads>>>(input, output, nx, ny, nz, stride_x, stride_yx);
      break;
    case 4:
      medfilt_kernel_3D_r4<<<blocks, threads>>>(input, output, nx, ny, nz, stride_x, stride_yx);
      break;
    case 5:
      medfilt_kernel_3D_r5<<<blocks, threads>>>(input, output, nx, ny, nz, stride_x, stride_yx);
      break;
    default:
      fprintf(stderr,"ERROR: medilter_kernel_3D not implemented for radius=%d\n", radius);
  }
}

////////////////////////////////////////////////
/////////////// HOST FUNCTION ///////////////////
/////////////////////////////////////////////////
extern "C" int MedianFilt_shared_GPU_main_float32(
  float *Input,   // outer input volume padded
  float *Output,  // outer output volume padded
  int radius,     // padding width
  int gpu_device, // requested device id
  int N, // outer volume padded faster dimension
  int M, 
  int Z  // outer volume padded slower dimension
)
{
  int deviceCount = -1; // number of devices
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    fprintf(stderr, "No CUDA devices found\n");
    return -1;
  }

  /*set GPU device*/
  if (deviceCount <= gpu_device) {
    fprintf(stderr, "Not enough devices to pick device id %d. Selecting last available device\n", gpu_device);
    gpu_device = deviceCount-1;
  }
  checkCudaErrors(cudaSetDevice(gpu_device));  

  float *d_input, *d_output;
  size_t ImSize = ((size_t) N) * M * Z;

  checkCudaErrors(cudaMalloc((void**)&d_input,ImSize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)&d_output,ImSize*sizeof(float)));

  checkCudaErrors(cudaMemcpy(d_input,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_output,Input,ImSize*sizeof(float),cudaMemcpyHostToDevice));

  // N,M,Z are outer dimensions (padded)
  // nx, ny, nz are inner dimensions
  int nx = N - 2*radius;
  int ny = M - 2*radius;
  int nz = Z - 2*radius;
  // pointing to first element of inner volumes inside padded outer volumes
  float *inner_input  = d_input  + radius*(1 + N + N*M);
  float *inner_output = d_output + radius*(1 + N + N*M);

  // medfilt_kernel_2D(inner_input, inner_output, nx, ny, nz, N, N*M, radius);
  medfilt_kernel_3D(inner_input, inner_output, nx, ny, nz, N, N*M, radius);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());

  checkCudaErrors(cudaMemcpy(Output,d_output,ImSize*sizeof(float),cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_input));
  checkCudaErrors(cudaFree(d_output));

  return 0;    
}