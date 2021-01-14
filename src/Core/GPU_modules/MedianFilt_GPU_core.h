#ifndef __MedFiltGPU_H__
#define __MedFiltGPU_H__
#include "DLSDefines.h"
#include <stdio.h>

extern "C" DLS_EXPORT int MedianFilt_GPU_main_float32(float *Input, float *Output, int kernel_size, float mu_threshold, int N, int M, int Z);
extern "C" DLS_EXPORT int MedianFilt_GPU_main_uint16(unsigned short *Input, unsigned short *Output, int kernel_size, float mu_threshold, int N, int M, int Z);

#endif
