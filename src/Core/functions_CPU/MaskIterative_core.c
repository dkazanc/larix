/*
 *
 * Copyright 2019 Daniil Kazantsev
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
 
void swap(float *xp, float *yp)
{
    float temp = *xp;
    *xp = *yp;
    *yp = temp;
}


float MASK_flat_main(float *Input, unsigned char *MASK_in, unsigned char *MASK_out, float threhsold, int iterations, int method, int dimX, int dimY, int dimZ)
{
    int i;
    float *maskreg_value;

    maskreg_value = (float*) calloc (2,sizeof(float));

    /* copy given MASK to MASK_out*/
    copyIm_unchar(MASK_in, MASK_out, (long)(dimX), (long)(dimY), (long)(dimZ));

    if (dimZ == 1) {
      /*2D version*/
      /* calculate mean inside given MASK */
      if (method == 0) mask_region_MADmean(Input, MASK_out, maskreg_value, (long)(dimX), (long)(dimY));
      if (method == 2) mask_region_MADmedian(Input, MASK_out, maskreg_value, (long)(dimX), (long)(dimY));

      /* iteratively updating mask */
      for(i=0; i<iterations; i++) {
        mask_update(Input, MASK_out, maskreg_value, threhsold, method, (long)(dimX), (long)(dimY));
      }
      /*printf("%f\n", maskreg_value[0]);*/
      }
    else {
    /*3D version*/
    /* calculate mean inside given MASK */
    if (method == 0) mask_region_MADmean3D(Input, MASK_out, maskreg_value, (long)(dimX), (long)(dimY), (long)(dimZ));
    if (method == 2) mask_region_MADmedian3D(Input, MASK_out, maskreg_value, (long)(dimX), (long)(dimY), (long)(dimZ));
    
    /* iteratively updating 3D mask */
    for(i=0; i<iterations; i++) {
      mask_update3D(Input, MASK_out, maskreg_value, threhsold, method, (long)(dimX), (long)(dimY), (long)(dimZ));
      }
    }
    free(maskreg_value);
    return *MASK_out;
}


/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/

float mask_region_MADmean(float *Input, unsigned char *MASK, float *maskreg_value, long dimX, long dimY)
{
    float mean_final, sum_residual;
    int index, j, i, counter;

    mean_final = 0.0f;
    sum_residual = 0.0f;
    counter = 0;
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = j*dimX+i;
        if (MASK[index] != 0) {
        mean_final += Input[index];
        counter++; }
    }}
    if (counter != 0) {
    mean_final /= counter;
    
    /* here we obtain the mean absolute deviation value within the mask */
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = j*dimX+i;
        if (MASK[index] != 0) sum_residual += fabs(Input[index] - mean_final);
    }}
    sum_residual /= counter;
    
    maskreg_value[0] = mean_final;
    maskreg_value[1] = 1.4826*sum_residual;
    }
    else {
    maskreg_value[0] = 0.0;
    maskreg_value[1] = 0.0;
    }
    return *maskreg_value;
}

float mask_region_MADmedian(float *Input, unsigned char *MASK, float *maskreg_value, long dimX, long dimY)
{
    float *Values_Vec, median_final, MAD_abs_final;
    int index, j, i, x, y, counter, midval;

    median_final = 0.0f;
    counter = 0;
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = j*dimX+i;
        if (MASK[index] != 0) {
        counter++; }
    }}
    
    Values_Vec = (float*) calloc (counter, sizeof(float));
    counter = 0;
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = j*dimX+i;
        if (MASK[index] != 0) {
        Values_Vec[counter] = Input[index];
        counter++; }
    }}
    
    midval = (int)(0.5f*counter) - 1;
    if (counter != 0) {
    /* perform sorting of the vector array */
    for (x = 0; x < counter-1; x++)  {
        for (y = 0; y < counter-x-1; y++)  {
            if (Values_Vec[y] > Values_Vec[y+1]) {
                swap(&Values_Vec[y], &Values_Vec[y+1]);
            }}}
    median_final = Values_Vec[midval];
    
    counter = 0;
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = j*dimX+i;
        if (MASK[index] != 0) {
        Values_Vec[counter] = fabs(Input[index] - median_final);
        counter++;}
    }}
    /* perform sorting of the vector array */
    for (x = 0; x < counter-1; x++)  {
        for (y = 0; y < counter-x-1; y++)  {
            if (Values_Vec[y] > Values_Vec[y+1]) {
                swap(&Values_Vec[y], &Values_Vec[y+1]);
            }}}
    MAD_abs_final = Values_Vec[midval];
    
    maskreg_value[0] = median_final;
    maskreg_value[1] = 1.4826*MAD_abs_final;
    }
    free(Values_Vec);
    return *maskreg_value;
}



float mask_update(float *Input, unsigned char *MASK, float *maskreg_value, float threhsold, int method, long dimX, long dimY)
{
    int index, j, i, i_s, i_n, j_e, j_w;

#pragma omp parallel for shared (Input, MASK, maskreg_value, method) private(index, j, i, i_s, i_n, j_e, j_w)
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
        if ((method == 0) || (method == 2)) {
            if (fabs(Input[index] - maskreg_value[0]) <=  threhsold*maskreg_value[1]) {
              /* make the central pixel part of the mask */
                  MASK[index] = 1;
                }
        }
        else {
          if (fabs(Input[index]) <=  threhsold) {
          /* make the central pixel part of the mask */
              MASK[index] = 1;
            }
        }
          }
        }
    }}
    return *MASK;
}

/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/
float mask_region_MADmean3D(float *Input, unsigned char *MASK, float *maskreg_value, long dimX, long dimY, long dimZ)
{
    float mean_final, sum_residual;
    int index, j, i, k, counter;

    mean_final = 0.0f;
    sum_residual = 0.0f;
    counter = 0;

  for(k=0; k<dimZ; k++) {
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = (dimX*dimY)*k + j*dimX+i;
        if (MASK[index] != 0) {
        mean_final += Input[index];
        counter++; }
    }}}
    if (counter != 0) {
    mean_final /= counter;
    /* here we obtain the mean absolute deviation value within the mask */
   for(k=0; k<dimZ; k++) {
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = (dimX*dimY)*k + j*dimX+i;
        if (MASK[index] != 0) sum_residual += fabs(Input[index] - mean_final);
    }}}
    sum_residual /= counter;
    
    maskreg_value[0] = mean_final;
    maskreg_value[1] = 1.4826*sum_residual;
    }
    else {
    maskreg_value[0] = 0.0;
    maskreg_value[1] = 0.0;
    }
    return *maskreg_value;
}


float mask_region_MADmedian3D(float *Input, unsigned char *MASK, float *maskreg_value, long dimX, long dimY, long dimZ)
{
    float *Values_Vec, median_final, MAD_abs_final;
    int index, j, i, k, x, y, counter, midval;

    median_final = 0.0f;
    counter = 0;
   for(k=0; k<dimZ; k++) {
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = (dimX*dimY)*k + j*dimX+i;
        if (MASK[index] != 0) {
        counter++; }
    }}}
    
    Values_Vec = (float*) calloc (counter, sizeof(float));
    counter = 0;
  for(k=0; k<dimZ; k++) {
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = (dimX*dimY)*k + j*dimX+i;
        if (MASK[index] != 0) {
        Values_Vec[counter] = Input[index];
        counter++; }
    }}}
    
    midval = (int)(0.5f*counter) - 1;
    if (counter != 0) {
    /* perform sorting of the vector array */
    for (x = 0; x < counter-1; x++)  {
        for (y = 0; y < counter-x-1; y++)  {
            if (Values_Vec[y] > Values_Vec[y+1]) {
                swap(&Values_Vec[y], &Values_Vec[y+1]);
            }}}
    median_final = Values_Vec[midval];
    
    counter = 0;
    for(k=0; k<dimZ; k++) {
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
            index = (dimX*dimY)*k + j*dimX+i;
        if (MASK[index] != 0) {
        Values_Vec[counter] = fabs(Input[index] - median_final);
        counter++;}
    }}}
    /* perform sorting of the vector array */
    for (x = 0; x < counter-1; x++)  {
        for (y = 0; y < counter-x-1; y++)  {
            if (Values_Vec[y] > Values_Vec[y+1]) {
                swap(&Values_Vec[y], &Values_Vec[y+1]);
            }}}
    MAD_abs_final = Values_Vec[midval];
    
    maskreg_value[0] = median_final;
    maskreg_value[1] = 1.4826*MAD_abs_final;
    }
    free(Values_Vec);
    return *maskreg_value;
}

float mask_update3D(float *Input, unsigned char *MASK, float *maskreg_value, float threhsold, int method, long dimX, long dimY, long dimZ)
{
    int index, j, i, k, i_s, i_n, j_e, j_w, k_u, k_d;

#pragma omp parallel for shared (Input, MASK, maskreg_value, method) private(index, j, i, k, i_s, i_n, j_e, j_w, k_u, k_d)
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
        if ((MASK[(dimX*dimY)*k + j*dimX+i_s] == 1) || (MASK[(dimX*dimY)*k + j*dimX+i_n] == 1) || (MASK[(dimX*dimY)*k + j_e*dimX+i] == 1) || (MASK[(dimX*dimY)*k + j_w*dimX+i] == 1) || (MASK[(dimX*dimY)*k_d + j*dimX+i] == 1) || (MASK[(dimX*dimY)*k_u + j*dimX+i] == 1)) {
        /* test the central pixel if it belongs to the same class */
        if ((method == 0) || (method == 2)) {
            if (fabs(Input[index] - maskreg_value[0]) <=  threhsold*maskreg_value[1]) {
              /* make the central pixel part of the mask */
                  MASK[index] = 1;
            }
        }
        else {
          if (fabs(Input[index]) <=  threhsold) {
          /* make the central pixel part of the mask */
              MASK[index] = 1;
                  }
        }
            }
        }
    }}}
    return *MASK;
}
