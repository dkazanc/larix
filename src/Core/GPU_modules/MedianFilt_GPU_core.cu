/* This software has been developed at Diamond Light Source Ltd.
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
/* CUDA implementation of the median filtration and dezingering (2D/3D case) 
 * using global memory with streaming (thanks to Yousef Moazzam)
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. radius: The half-size (radius) of the median filter window
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
      //quicksort_float(ValVec, 0, diameter*diameter); /* perform sorting */
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


/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/
template <int radius, int diameter, int midpoint> // diameter should be set to 2*radius+1
inline __device__ void medfilt_kernel_3D_global_float_t(
    float *Input,
    float *Output,
    unsigned long long offset,
    int N, // faster dimension
    int M,
    int Z,
    unsigned long long num_total,
    float mu_threshold
    )
{  
      float ValVec[diameter*diameter*diameter];
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
      for(i_m=-radius; i_m<=radius; i_m++) {
            i1 = i + i_m;
            if ((i1 < 0) || (i1 >= N)) i1 = i;
            for(j_m=-radius; j_m<=radius; j_m++) {
              j1 = j + j_m;
              if ((j1 < 0) || (j1 >= M)) j1 = j;
                for(k_m=-radius; k_m<=radius; k_m++) {
                  k1 = k + k_m;
                  if ((k1 < 0) || (k1 >= Z)) k1 = k;
                  ValVec[counter] = Input[i1 + N*j1 + N*M*k1];
                  counter++;
      }}}
      //sort_quick(ValVec, 0, diameter*diameter*diameter); /* perform sorting */
      sort_bubble(ValVec, diameter*diameter*diameter); /* perform sorting */

      if (mu_threshold == 0.0f) Output[index] = ValVec[midpoint]; /* perform median filtration */
      else {
      /* perform dezingering */
      if (abs(Input[index] - ValVec[midpoint]) >= mu_threshold) Output[index] = ValVec[midpoint];
        }
      }
      return;
}

// instances of predefined kernel templates with selected radius/diameter
__global__ void medfilt_kernel_global_3D_r1(float *Input, float *Output, unsigned long long offset, int N, int M, int Z, unsigned long long num_total, float mu_threshold)
{
  medfilt_kernel_3D_global_float_t<1,3,13>(Input, Output, offset, N, M, Z, num_total, mu_threshold);
}
__global__ void medfilt_kernel_global_3D_r2(float *Input, float *Output, unsigned long long offset, int N, int M, int Z, unsigned long long num_total, float mu_threshold)
{
  medfilt_kernel_3D_global_float_t<2,5,62>(Input, Output, offset, N, M, Z, num_total, mu_threshold);
}
__global__ void medfilt_kernel_global_3D_r3(float *Input, float *Output, unsigned long long offset, int N, int M, int Z, unsigned long long num_total, float mu_threshold)
{
  medfilt_kernel_3D_global_float_t<3,7,171>(Input, Output, offset, N, M, Z, num_total, mu_threshold);
}
__global__ void medfilt_kernel_global_3D_r4(float *Input, float *Output, unsigned long long offset, int N, int M, int Z, unsigned long long num_total, float mu_threshold)
{
  medfilt_kernel_3D_global_float_t<4,9,364>(Input, Output, offset, N, M, Z, num_total, mu_threshold);
}
__global__ void medfilt_kernel_global_3D_r5(float *Input, float *Output, unsigned long long offset, int N, int M, int Z, unsigned long long num_total, float mu_threshold)
{
  medfilt_kernel_3D_global_float_t<5,11,665>(Input, Output, offset, N, M, Z, num_total, mu_threshold);
}

/////////////////////////////////////////////////
/////////////// HOST FUNCTION ///////////////////
/////////////////////////////////////////////////
extern "C" int MedianFilt_global_GPU_main_float32(float *Input, float *Output, int radius, float mu_threshold, int gpu_device, int N, int M, int Z)
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

        const unsigned long long ImSize = (unsigned long long)N*(unsigned long long)M*(unsigned long long)Z;
        float *d_input0, *d_output0;

        checkCudaErrors(cudaSetDevice(gpu_device)); /*set the GPU device*/
        const int nStreams = 4; /* set the streams number */
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

	if (Z == 1) {
        /*2D case */
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
           switch (radius) {
            case 1:
              medfilt_kernel_global_2D_r1<<<dimGrid, dimBlock, 0, stream[i]>>>(d_input0, d_output0, process_offset, N, M, (int)(ImSize), mu_threshold);
              break;
            case 2:
              medfilt_kernel_global_2D_r2<<<dimGrid, dimBlock, 0, stream[i]>>>(d_input0, d_output0, process_offset, N, M, (int)(ImSize), mu_threshold);
              break;
            case 3:
              medfilt_kernel_global_2D_r3<<<dimGrid, dimBlock, 0, stream[i]>>>(d_input0, d_output0, process_offset, N, M, (int)(ImSize), mu_threshold);
              break;
            case 4:
              medfilt_kernel_global_2D_r4<<<dimGrid, dimBlock, 0, stream[i]>>>(d_input0, d_output0, process_offset, N, M, (int)(ImSize), mu_threshold);
              break;
            case 5:
              medfilt_kernel_global_2D_r5<<<dimGrid, dimBlock, 0, stream[i]>>>(d_input0, d_output0, process_offset, N, M, (int)(ImSize), mu_threshold);
              break;
            default:
              fprintf(stderr,"ERROR: medilter_kernel_2D not implemented for radius=%d\n", radius);
          }
          /* copy processed data from device to the host */
          checkCudaErrors( cudaMemcpyAsync(&pinned_mem_output[copy_offset_d2h], &d_output0[copy_offset_d2h],
                                     bytes_to_copy_d2h, cudaMemcpyDeviceToHost,
                                     stream[i]) );
          checkCudaErrors( cudaDeviceSynchronize() );
          }
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
        
        // number of bytes to copy for a stream
        unsigned long long bytes_to_copy_h2d;
        unsigned long long bytes_to_copy_d2h;
        unsigned long long copy_offset_h2d;
        unsigned long long copy_offset_d2h;
        unsigned long long process_offset;

        // WITH streaming, looping over different operations
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

          // running the kernel
           switch (radius) {
            case 1:
              medfilt_kernel_global_3D_r1<<<dimGrid, dimBlock, 0, stream[i]>>>(d_input0, d_output0, process_offset, N, M, Z, ImSize, mu_threshold);
              break;
            case 2:
              medfilt_kernel_global_3D_r2<<<dimGrid, dimBlock, 0, stream[i]>>>(d_input0, d_output0, process_offset, N, M, Z, ImSize, mu_threshold);
              break;
            case 3:
              medfilt_kernel_global_3D_r3<<<dimGrid, dimBlock, 0, stream[i]>>>(d_input0, d_output0, process_offset, N, M, Z, ImSize, mu_threshold);
              break;
            case 4:
              medfilt_kernel_global_3D_r4<<<dimGrid, dimBlock, 0, stream[i]>>>(d_input0, d_output0, process_offset, N, M, Z, ImSize, mu_threshold);
              break;
            case 5:
              medfilt_kernel_global_3D_r5<<<dimGrid, dimBlock, 0, stream[i]>>>(d_input0, d_output0, process_offset, N, M, Z, ImSize, mu_threshold);
              break;
            default:
              fprintf(stderr,"ERROR: medfilter_kernel_3D is not implemented for radius=%d\n", radius);
          }
            // copy a slab of the processed data from GPU to CPU memory
            checkCudaErrors( cudaMemcpyAsync(&pinned_mem_output[copy_offset_d2h], &d_output0[copy_offset_d2h],
                                      bytes_to_copy_d2h, cudaMemcpyDeviceToHost,
                                      stream[i]) );
            checkCudaErrors( cudaDeviceSynchronize() );
            checkCudaErrors(cudaPeekAtLastError() );                                      
          }
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
        return 0;
}