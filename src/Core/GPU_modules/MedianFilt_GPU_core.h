#ifndef __MedFiltGPU_H__
#define __MedFiltGPU_H__
#include "DLSDefines.h"
#include <stdio.h>

extern "C" DLS_EXPORT int MedianFilt_GPU_main(float *Input, float *Output, int filter_half_window_size, float mu_threshold, int N, int M, int Z);

#endif
