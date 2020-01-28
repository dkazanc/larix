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
 *
 * method 0 - based on the given intensity threshold only
 * method 1 - based on the mean calculated in the mask and Mean Absolute deviation for thresholding
 * method 2 - based on the median calculated in the mask and Median Absolute deviation for thresholding
 * Output:
 [1] Updated (evolved) mask (segmentation)
 */

void swap(float *xp, float *yp)
{
    float temp = *xp;
    *xp = *yp;
    *yp = temp;
}

float MASK_evolve_main(float *Input, unsigned char *MASK_in, unsigned char *MASK_out, float threhsold, int iterations, int connectivity, int method, int dimX, int dimY, int dimZ)
{
    int i;
    float *maskreg_value;

    maskreg_value = (float*) calloc (2,sizeof(float));

    /* copy given MASK to MASK_out*/
    copyIm_unchar(MASK_in, MASK_out, (long)(dimX), (long)(dimY), (long)(dimZ));

    if (dimZ == 1) {
      /*2D version*/
      /* calculate mean inside given MASK */
      if (method == 1) mask_region_MADmean(Input, MASK_out, maskreg_value, (long)(dimX), (long)(dimY));
      if (method == 2) mask_region_MADmedian(Input, MASK_out, maskreg_value, (long)(dimX), (long)(dimY));

      /* iteratively updating the 2D mask */
      for(i=0; i<iterations; i++) {
        if (connectivity == 8) mask_update8(Input, MASK_out, maskreg_value, threhsold, method, (long)(dimX), (long)(dimY));
        else mask_update4(Input, MASK_out, maskreg_value, threhsold, method, (long)(dimX), (long)(dimY));
        }
      }
    else {
    /* 3D version */
    /* calculate mean inside given MASK */
    if (method == 1) mask_region_MADmean3D(Input, MASK_out, maskreg_value, (long)(dimX), (long)(dimY), (long)(dimZ));
    if (method == 2) mask_region_MADmedian3D(Input, MASK_out, maskreg_value, (long)(dimX), (long)(dimY), (long)(dimZ));
    //printf("%f\n", maskreg_value[0]);

    /* iteratively updating the 3D mask */
    for(i=0; i<iterations; i++) {
        if (connectivity == 4) {
          /* connectivity of 4 pixels in 2D space */
          mask_update3D_4(Input, MASK_out, maskreg_value, threhsold, method, (long)(dimX), (long)(dimY), (long)(dimZ)); }
        else if (connectivity == 8) {
          /* connectivity of 8 pixels in 2D space */
          mask_update3D_8(Input, MASK_out, maskreg_value, threhsold, method, (long)(dimX), (long)(dimY), (long)(dimZ));}
        else if (connectivity == 26) {
          /* connectivity of 26 pixels in 3D space */
          mask_update3D_26(Input, MASK_out, maskreg_value, threhsold, method, (long)(dimX), (long)(dimY), (long)(dimZ));}
        else {
          /* by default the connectivity of 6 pixels in 3D space */
          mask_update3D_6(Input, MASK_out, maskreg_value, threhsold, method, (long)(dimX), (long)(dimY), (long)(dimZ));}
       }
     }
    free(maskreg_value);
    return *MASK_out;
}

float MASK_evolve_conditional_main(float *Input, unsigned char *MASK_in, unsigned char *MASK_conditional, unsigned char *MASK_out, float threhsold, int iterations, int connectivity, int method, int dimX, int dimY, int dimZ)
{
    int i;
    float *maskreg_value;

    maskreg_value = (float*) calloc (2,sizeof(float));

    /* copy given MASK to MASK_out*/
    copyIm_unchar(MASK_in, MASK_out, (long)(dimX), (long)(dimY), (long)(dimZ));

    if (dimZ == 1) {
      /*2D version*/
      /* calculate mean inside given MASK */
      if (method == 1) mask_region_MADmean(Input, MASK_out, maskreg_value, (long)(dimX), (long)(dimY));
      if (method == 2) mask_region_MADmedian(Input, MASK_out, maskreg_value, (long)(dimX), (long)(dimY));

      /* iteratively updating the 2D mask */
      for(i=0; i<iterations; i++) {
        if (connectivity == 8) mask_update_con8(Input, MASK_conditional, MASK_out, maskreg_value, threhsold, method, (long)(dimX), (long)(dimY));
        else mask_update_con4(Input, MASK_conditional, MASK_out, maskreg_value, threhsold, method, (long)(dimX), (long)(dimY));
        }
      }
    else {
    /* 3D version */
    /* calculate mean inside given MASK */
    if (method == 1) mask_region_MADmean3D(Input, MASK_out, maskreg_value, (long)(dimX), (long)(dimY), (long)(dimZ));
    if (method == 2) mask_region_MADmedian3D(Input, MASK_out, maskreg_value, (long)(dimX), (long)(dimY), (long)(dimZ));
    //printf("%f\n", maskreg_value[0]);

    /* iteratively updating the 3D mask */
    for(i=0; i<iterations; i++) {
        if (connectivity == 4) {
          /* connectivity of 4 pixels in 2D space */
          mask_update_con3D_4(Input, MASK_conditional, MASK_out, maskreg_value, threhsold, method, (long)(dimX), (long)(dimY), (long)(dimZ)); }
        else if (connectivity == 8) {
          /* connectivity of 8 pixels in 2D space */
          mask_update_con3D_8(Input, MASK_conditional, MASK_out, maskreg_value, threhsold, method, (long)(dimX), (long)(dimY), (long)(dimZ));}
        else if (connectivity == 26) {
          /* connectivity of 26 pixels in 3D space */
          mask_update_con3D_26(Input, MASK_conditional, MASK_out, maskreg_value, threhsold, method, (long)(dimX), (long)(dimY), (long)(dimZ));}
        else {
          /* by default the connectivity of 6 pixels in 3D space */
          mask_update_con3D_6(Input, MASK_conditional, MASK_out, maskreg_value, threhsold, method, (long)(dimX), (long)(dimY), (long)(dimZ));}
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

/* connectivity of 4 pixels */
float mask_update4(float *Input, unsigned char *MASK, float *maskreg_value, float threhsold, int method, long dimX, long dimY)
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
        /* find if the closest pixels of the mask are equal to 1 */
        if ((MASK[j*dimX+i_s] == 1) || (MASK[j*dimX+i_n] == 1) || (MASK[j_e*dimX+i] == 1) || (MASK[j_w*dimX+i] == 1)) {
        /* test if the central pixel also belongs to the same class  as the neighbourhood*/
        if ((method == 1) || (method == 2)) {
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
/* connectivity of 8 pixels */
float mask_update8(float *Input, unsigned char *MASK, float *maskreg_value, float threhsold, int method, long dimX, long dimY)
{
    int index, j, i, i1, j1, i_m, j_m;

#pragma omp parallel for shared (Input, MASK, maskreg_value, method) private(index, j, i, i1, j1, i_m, j_m)
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = j*dimX+i;

          for(j_m=-1; j_m<=1; j_m++) {
            for(i_m=-1; i_m<=1; i_m++) {
                i1 = i+i_m;
                j1 = j+j_m;
                if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {

              /* find if the closest pixels of the mask are equal to 1 */
               if (MASK[j1*dimX+i1] == 1) {
              /* test if the central pixel also belongs to the same class  as the neighbourhood*/
              if ((method == 1) || (method == 2)) {
                  if (fabs(Input[index] - maskreg_value[0]) <=  threhsold*maskreg_value[1]) {
              /* make the central pixel part of the mask */
                  MASK[index] = 1;  }
              else {
              if (fabs(Input[index]) <=  threhsold) {
              /* make the central pixel part of the mask */
                  MASK[index] = 1;  }
                }
            } /*((method == 1) || (method == 2))*/
         }
       }
     }}
    }}
    return *MASK;
}


/* mask conditional  connectivity of 4 pixels */
float mask_update_con4(float *Input, unsigned char *MASK_conditional, unsigned char *MASK, float *maskreg_value, float threhsold, int method, long dimX, long dimY)
{
    int index, j, i, i_s, i_n, j_e, j_w;

#pragma omp parallel for shared (Input, MASK, MASK_conditional, maskreg_value, method) private(index, j, i, i_s, i_n, j_e, j_w)
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        i_s = i + 1;
        i_n = i - 1;
        j_e = j + 1;
        j_w = j - 1;
        index = j*dimX+i;

        if (MASK_conditional[index] == 0) {
        if (((i_n >= 0) && (i_s < dimX)) && ((j_w >= 0) && (j_e < dimY))) {
        /* find if the closest pixels of the mask are equal to 1 */
        if ((MASK[j*dimX+i_s] == 1) || (MASK[j*dimX+i_n] == 1) || (MASK[j_e*dimX+i] == 1) || (MASK[j_w*dimX+i] == 1)) {
        /* test if the central pixel also belongs to the same class  as the neighbourhood*/
        if ((method == 1) || (method == 2)) {
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
      }
    }}
    return *MASK;
}
/* mask conditional connectivity of 8 pixels */
float mask_update_con8(float *Input, unsigned char *MASK_conditional,  unsigned char *MASK, float *maskreg_value, float threhsold, int method, long dimX, long dimY)
{
    int index, j, i, i1, j1, i_m, j_m;

#pragma omp parallel for shared (Input, MASK, MASK_conditional, maskreg_value, method) private(index, j, i, i1, j1, i_m, j_m)
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = j*dimX+i;

        if (MASK_conditional[index] == 0) {
          for(j_m=-1; j_m<=1; j_m++) {
            for(i_m=-1; i_m<=1; i_m++) {
                i1 = i+i_m;
                j1 = j+j_m;
                if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {

              /* find if the closest pixels of the mask are equal to 1 */
               if (MASK[j1*dimX+i1] == 1) {
              /* test if the central pixel also belongs to the same class  as the neighbourhood*/
              if ((method == 1) || (method == 2)) {
                  if (fabs(Input[index] - maskreg_value[0]) <=  threhsold*maskreg_value[1]) {
              /* make the central pixel part of the mask */
                  MASK[index] = 1;  }
              else {
              if (fabs(Input[index]) <=  threhsold) {
              /* make the central pixel part of the mask */
                  MASK[index] = 1;  }
                }
            } /*((method == 1) || (method == 2))*/
          }
        }
      }}
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

float mask_update3D_4(float *Input, unsigned char *MASK, float *maskreg_value, float threhsold, int method, long dimX, long dimY, long dimZ)
{
    int index, j, i, k, i_s, i_n, j_e, j_w;

#pragma omp parallel for shared (Input, MASK, maskreg_value, method) private(index, j, i, k, i_s, i_n, j_e, j_w)
    for(k=0; k<dimZ; k++) {
      for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        i_s = i + 1;
        i_n = i - 1;
        j_e = j + 1;
        j_w = j - 1;
        index = (dimX*dimY)*k + j*dimX+i;

        if (((i_n >= 0) && (i_s < dimX)) && ((j_w >= 0) && (j_e < dimY))) {
          /* find if the closest pixels of the mask are equal to 1 */
        if ((MASK[(dimX*dimY)*k + j*dimX+i_s] == 1) || (MASK[(dimX*dimY)*k + j*dimX+i_n] == 1) || (MASK[(dimX*dimY)*k + j_e*dimX+i] == 1) || (MASK[(dimX*dimY)*k + j_w*dimX+i] == 1)) {
        /* test if the central pixel also belongs to the same class  as the neighbourhood*/
        if ((method == 1) || (method == 2)) {
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

float mask_update3D_6(float *Input, unsigned char *MASK, float *maskreg_value, float threhsold, int method, long dimX, long dimY, long dimZ)
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
        /* find if the closest pixels of the mask are equal to 1 */
        if ((MASK[(dimX*dimY)*k + j*dimX+i_s] == 1) || (MASK[(dimX*dimY)*k + j*dimX+i_n] == 1) || (MASK[(dimX*dimY)*k + j_e*dimX+i] == 1) || (MASK[(dimX*dimY)*k + j_w*dimX+i] == 1) || (MASK[(dimX*dimY)*k_d + j*dimX+i] == 1) || (MASK[(dimX*dimY)*k_u + j*dimX+i] == 1)) {
        /* test if the central pixel also belongs to the same class  as the neighbourhood*/
        if ((method == 1) || (method == 2)) {
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

float mask_update3D_8(float *Input, unsigned char *MASK, float *maskreg_value, float threhsold, int method, long dimX, long dimY, long dimZ)
{
    int index, j, i, k, i_m, j_m, i1, j1;

#pragma omp parallel for shared (Input, MASK, maskreg_value, method) private(index, j, i, k, i_m, j_m, i1, j1)
    for(k=0; k<dimZ; k++) {
      for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = (dimX*dimY)*k + j*dimX+i;

          for(j_m=-1; j_m<=1; j_m++) {
            for(i_m=-1; i_m<=1; i_m++) {
              i1 = i+i_m;
              j1 = j+j_m;

              if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {

                /* find if the closest pixels of the mask are equal to 1 */
                if (MASK[(dimX*dimY)*k + j1*dimX+i1] == 1) {
                /* test if the central pixel also belongs to the same class  as the neighbourhood*/
                if ((method == 1) || (method == 2)) {
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
    }}}
    return *MASK;
}



float mask_update3D_26(float *Input, unsigned char *MASK, float *maskreg_value, float threhsold, int method, long dimX, long dimY, long dimZ)
{
    int index, j, i, k, i_m, j_m, k_m, i1, j1, k1;

#pragma omp parallel for shared (Input, MASK, maskreg_value, method) private(index, j, i, k, i_m, j_m, k_m, i1, j1, k1)
    for(k=0; k<dimZ; k++) {
      for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = (dimX*dimY)*k + j*dimX+i;

        for(k_m=-1; k_m<=1; k_m++) {
          for(j_m=-1; j_m<=1; j_m++) {
            for(i_m=-1; i_m<=1; i_m++) {

              i1 = i+i_m;
              j1 = j+j_m;
              k1 = k+k_m;
              if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY)) && ((k1 >= 0) && (k1 < dimZ))) {
              /* find if the closest pixels of the mask are equal to 1 */
              if (MASK[(dimX*dimY)*k1 + j1*dimX+i1] == 1) {
                /* test if the central pixel also belongs to the same class as the neighbourhood*/
                if ((method == 1) || (method == 2)) {
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
    }}}
    return *MASK;
}

/* mask conditioned modules */
/*************************************************************************/
float mask_update_con3D_4(float *Input, unsigned char *MASK_conditional, unsigned char *MASK, float *maskreg_value, float threhsold, int method, long dimX, long dimY, long dimZ)
{
    int index, j, i, k, i_s, i_n, j_e, j_w;

#pragma omp parallel for shared (Input, MASK, MASK_conditional, maskreg_value, method) private(index, j, i, k, i_s, i_n, j_e, j_w)
    for(k=0; k<dimZ; k++) {
      for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        i_s = i + 1;
        i_n = i - 1;
        j_e = j + 1;
        j_w = j - 1;
        index = (dimX*dimY)*k + j*dimX+i;

        if (MASK_conditional[index] == 0) {
        if (((i_n >= 0) && (i_s < dimX)) && ((j_w >= 0) && (j_e < dimY))) {
          /* find if the closest pixels of the mask are equal to 1 */
        if ((MASK[(dimX*dimY)*k + j*dimX+i_s] == 1) || (MASK[(dimX*dimY)*k + j*dimX+i_n] == 1) || (MASK[(dimX*dimY)*k + j_e*dimX+i] == 1) || (MASK[(dimX*dimY)*k + j_w*dimX+i] == 1)) {
        /* test if the central pixel also belongs to the same class  as the neighbourhood*/
        if ((method == 1) || (method == 2)) {
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
      }
    }}}
    return *MASK;
}

float mask_update_con3D_6(float *Input, unsigned char *MASK_conditional, unsigned char *MASK, float *maskreg_value, float threhsold, int method, long dimX, long dimY, long dimZ)
{
    int index, j, i, k, i_s, i_n, j_e, j_w, k_u, k_d;
    int i_m, j_m, k_m, i1, j1, k1, counter;

#pragma omp parallel for shared (Input, MASK, MASK_conditional, maskreg_value, method) private(index, j, i, k, i_s, i_n, j_e, j_w, k_u, k_d, i_m, j_m, k_m, i1, j1, k1, counter)
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

        if (MASK_conditional[index] == 0) {
        if (((i_n >= 0) && (i_s < dimX)) && ((j_w >= 0) && (j_e < dimY)) && ((k_d >= 0) && (k_u < dimZ))) {
        /* find if the closest pixels of the mask are equal to 1 */
        if ((MASK[(dimX*dimY)*k + j*dimX+i_s] == 1) || (MASK[(dimX*dimY)*k + j*dimX+i_n] == 1) || (MASK[(dimX*dimY)*k + j_e*dimX+i] == 1) || (MASK[(dimX*dimY)*k + j_w*dimX+i] == 1) || (MASK[(dimX*dimY)*k_d + j*dimX+i] == 1) || (MASK[(dimX*dimY)*k_u + j*dimX+i] == 1)) {
        /* test if the central pixel also belongs to the same class  as the neighbourhood*/
        if ((method == 1) || (method == 2)) {
		
		counter = 0;
        for(k_m=-1; k_m<=1; k_m++) {
          for(j_m=-1; j_m<=1; j_m++) {
            for(i_m=-1; i_m<=1; i_m++) {
              i1 = i+i_m;
              j1 = j+j_m;
              k1 = k+k_m;
              if (MASK_conditional[(dimX*dimY)*k1 + j1*dimX+i1] == 1) counter++;
		  }}}

            if ((fabs(Input[index] - maskreg_value[0]) <=  threhsold*maskreg_value[1]) || (counter == 0)) {
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
      }
    }}}
    return *MASK;
}

float mask_update_con3D_8(float *Input, unsigned char *MASK_conditional, unsigned char *MASK, float *maskreg_value, float threhsold, int method, long dimX, long dimY, long dimZ)
{
    int index, j, i, k, i_m, j_m, i1, j1;

#pragma omp parallel for shared (Input, MASK, MASK_conditional, maskreg_value, method) private(index, j, i, k, i_m, j_m, i1, j1)
    for(k=0; k<dimZ; k++) {
      for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = (dimX*dimY)*k + j*dimX+i;

        if (MASK_conditional[index] == 0) {
          for(j_m=-1; j_m<=1; j_m++) {
            for(i_m=-1; i_m<=1; i_m++) {
              i1 = i+i_m;
              j1 = j+j_m;

              if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {

                /* find if the closest pixels of the mask are equal to 1 */
                if (MASK[(dimX*dimY)*k + j1*dimX+i1] == 1) {
                /* test if the central pixel also belongs to the same class  as the neighbourhood*/
                if ((method == 1) || (method == 2)) {
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
      }
    }}}
    return *MASK;
}



float mask_update_con3D_26(float *Input, unsigned char *MASK_conditional,  unsigned char *MASK, float *maskreg_value, float threhsold, int method, long dimX, long dimY, long dimZ)
{
    int index, j, i, k, i_m, j_m, k_m, i1, j1, k1;

#pragma omp parallel for shared (Input, MASK, MASK_conditional, maskreg_value, method) private(index, j, i, k, i_m, j_m, k_m, i1, j1, k1)
    for(k=0; k<dimZ; k++) {
      for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = (dimX*dimY)*k + j*dimX+i;

        if (MASK_conditional[index] == 0) {

        for(k_m=-1; k_m<=1; k_m++) {
          for(j_m=-1; j_m<=1; j_m++) {
            for(i_m=-1; i_m<=1; i_m++) {

              i1 = i+i_m;
              j1 = j+j_m;
              k1 = k+k_m;
              if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY)) && ((k1 >= 0) && (k1 < dimZ))) {
              /* find if the closest pixels of the mask are equal to 1 */
              if (MASK[(dimX*dimY)*k1 + j1*dimX+i1] == 1) {
                /* test if the central pixel also belongs to the same class as the neighbourhood*/
                if ((method == 1) || (method == 2)) {
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
      }
    }}}
    return *MASK;
}
