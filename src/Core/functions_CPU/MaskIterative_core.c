/*
 * This work is part of the Core Imaging Library developed by
 * Visual Analytics and Imaging System Group of the Science Technology
 * Facilities Council, STFC
 *
 * Copyright 2017 Daniil Kazantsev
 * Copyright 2017 Srikanth Nagella, Edoardo Pasca
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


#include "MaskIterative_core.h"
#include "utils.h"

/* C-OMP implementation of segmentation flattening-based process using the provided MASK.

 * Input Parameters:
 * 1. Noisy image/volume
 * 2. MASK (in unsigned short format)

 * Output:
 * [1] Updated mask (segmentation)
 *
 */

float MASK_flat_main(float *Input, unsigned char *MASK_in, unsigned char *MASK_out, float threhsold, int iterations, int dimX, int dimY, int dimZ)
{
    int i;
    float *maskreg_mean;

    maskreg_mean = (float*) calloc (1,sizeof(float));

    /* copy given MASK to MASK_out*/
    copyIm_unchar(MASK_in, MASK_out, (long)(dimX), (long)(dimY), (long)(dimZ));

    if (dimZ == 1) {
      /*2D version*/
      /* calculate mean inside given MASK */
      mask_region_mean(Input, MASK_out, maskreg_mean, (long)(dimX), (long)(dimY));

      /* iteratively updating mask */
      for(i=0; i<iterations; i++) {
        mask_update(Input, MASK_out, maskreg_mean, threhsold, (long)(dimX), (long)(dimY));
      }
    //printf("%f\n", maskreg_mean[0]);
      }
    else {
    /*3D version*/
    /* iteratively updating 3D mask */
    for(i=0; i<iterations; i++) {
      mask_update3D(Input, MASK_out, threhsold, (long)(dimX), (long)(dimY), (long)(dimZ));
      }
    }

    free(maskreg_mean);
    return *MASK_out;
}


/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
float mask_region_mean(float *Input, unsigned char *MASK, float *maskreg_mean, long dimX, long dimY)
{
    float mean_final, cur_val;
    int index, j, i, counter;

    mean_final = 0.0f;
    counter = 0;

    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = j*dimX+i;
        cur_val = Input[index]*(float)(MASK[index]);
        if (cur_val != 0.0f) {
        mean_final += cur_val;
        counter++; }
    }}
    if (counter != 0) {
        maskreg_mean[0] = mean_final/counter;
    }
    else maskreg_mean[0] = 0.0f;
    return *maskreg_mean;
}

float mask_update(float *Input, unsigned char *MASK, float *maskreg_mean, float threhsold, long dimX, long dimY)
{
    int index, j, i, i_s, i_n, j_e, j_w;

    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        i_s = i + 1;
        i_n = i - 1;
        j_e = j + 1;
        j_w = j - 1;
        index = j*dimX+i;

        if (((i_n >= 0) && (i_s < dimX)) && ((j_w >= 0) && (j_e < dimY))) {
        /* find where closest pixels of the mask equal to 1 */
        if ((MASK[j*dimX+i_s] == 1) || (MASK[j*dimX+i_n] == 1) || (MASK[j_e*dimX+i] == 1) || (MASK[j_w*dimX+i] == 1)) {
        /* test the central pixel if it belongs to the same class */
        // if (fabs(Input[index] - maskreg_mean[0]) <=  threhsold) {
        if (fabs(Input[index]) <=  threhsold) {
        /* make the central pixel part of the mask */
            MASK[index] = 1;
                }
            }
        }
    }}
    return *MASK;
}

/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/
float mask_update3D(float *Input, unsigned char *MASK, float threhsold, long dimX, long dimY, long dimZ)
{
    int index, j, i, k, i_s, i_n, j_e, j_w, k_u, k_d;

    for(k=0; k<dimZ; k++) {
      for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        i_s = i + 1;
        i_n = i - 1;
        j_e = j + 1;
        j_w = j - 1;
        k_u = k + 1;
        k_d = k - 1;
        index = (dimX*dimY)*k + j*dimX+i;

        if (((i_n >= 0) && (i_s < dimX)) && ((j_w >= 0) && (j_e < dimY)) && ((k_d >= 0) && (k_u < dimZ))) {
        /* find where closest pixels of the mask equal to 1 */
        if ((MASK[(dimX*dimY)*k + j*dimX+i_s] == 1) || (MASK[(dimX*dimY)*k + j*dimX+i_n] == 1) || (MASK[(dimX*dimY)*k + j_i*dimX+i] == 1) || (MASK[(dimX*dimY)*k + j_w*dimX+i] == 1) || (MASK[(dimX*dimY)*k_d + j*dimX+i] == 1) || (MASK[(dimX*dimY)*k_u + j*dimX+i] == 1)) {
        /* test the central pixel if it belongs to the same class */
        // if (fabs(Input[index] - maskreg_mean[0]) <=  threhsold) {
        if (fabs(Input[index]) <=  threhsold) {
        /* make the central pixel part of the mask */
            MASK[index] = 1;
                }
            }
        }
    }}}
    return *MASK;
}
