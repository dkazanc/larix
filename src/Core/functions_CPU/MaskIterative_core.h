/*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2019 Daniil Kazantsev

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/


#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"
#include "utils.h"

 
#ifdef __cplusplus
extern "C" {
#endif
float MASK_flat_main(float *Input, unsigned char *MASK_in, unsigned char *MASK_out, float threhsold, int iterations, int dimX, int dimY, int dimZ);
float mask_region_mean(float *Input, unsigned char *MASK, float *maskreg_mean, long dimX, long dimY);
/************2D functions ***********/
#ifdef __cplusplus
}
#endif
