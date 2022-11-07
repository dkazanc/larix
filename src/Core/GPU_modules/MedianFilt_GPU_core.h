#ifndef __MedFiltGPU_H__
#define __MedFiltGPU_H__
#include "DLSDefines.h"
#include <stdio.h>

extern "C" DLS_EXPORT int MedianFilt_global_GPU_main_float32(float *Input, float *Output, int radius, float mu_threshold, int gpu_device, int N, int M, int Z);

#endif
