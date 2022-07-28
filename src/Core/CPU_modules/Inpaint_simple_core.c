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

#include "Inpaint_simple_core.h"
#include "utils.h"

/* C-OMP implementation of simple inpainting shemes
 * inpainting using averaged interface values
 *
 * Input Parameters:
 * 1. Image/volume to inpaint
 * 2. Mask of the same size as (1) in 'unsigned char' format  (ones mark the region to inpaint, zeros belong to the data)
 * 3. Iterations number
 * 4. sigma - controlling parameter to start inpainting
 *
 * Output:
 * [1] Inpainted image/volume
 */

int Inpaint_simple_CPU_main(float *Input, unsigned char *Mask, float *Output, unsigned char *M_upd, int iterations, int W_halfsize, float sigma, int dimX, int dimY, int dimZ)
{
    long i, j, k, i1, j1, k1, l, countmask, DimTotal;
    int i_m, j_m;
    float *minmax_array, *Gauss_weights, *Updated=NULL, sumweigths;
    int W_fullsize, counter;

    DimTotal = (long)(dimX*dimY*dimZ);
    Updated = calloc(DimTotal, sizeof(float));

    /* copy input into output */
    copyIm(Input, Output, (long)(dimX), (long)(dimY), (long)(dimZ));
    copyIm(Input, Updated, (long)(dimX), (long)(dimY), (long)(dimZ));
    /* copying M to Mask_upd */
    copyIm_unchar(Mask, M_upd, dimX, dimY, dimZ);

    minmax_array = (float*) calloc (2,sizeof(float));
    max_val_mask(Input, M_upd, minmax_array, (long)(dimX), (long)(dimY), (long)(dimZ));

    /*calculate all nonzero values in the mask */
    countmask = 0;
    for (k=0; k<dimY*dimX*dimZ; k++) {
      if (M_upd[k] == 1) countmask++;
    }

    /* pre-calculation of Gaussian distance weights  */
    W_fullsize = (int)(2*W_halfsize + 1); /*full size of similarity window */
    Gauss_weights = (float*)calloc(W_fullsize*W_fullsize,sizeof(float ));
    sumweigths = 0.0f;
    counter = 0;
    for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) {
        for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) {
            Gauss_weights[counter] = expf(-(powf((i_m), 2) + powf((j_m), 2))/(2*W_halfsize*W_halfsize));
            sumweigths += Gauss_weights[counter];
            counter++;
        }
    }

    if (countmask == 0) {
      /*printf("%s \n", "Nothing to inpaint, zero mask!");*/
      free(minmax_array);
      free(Updated);
      free(Gauss_weights);
      return 0;
      }
    else {
    if (dimZ == 1) {
    for (l=0; l<iterations; l++) {
    #pragma omp parallel for shared(Input,M_upd,Gauss_weights) private(i1,j1)
    for(i1=0; i1<dimX; i1++) {
        for(j1=0; j1<dimY; j1++) {
    /*mean_inp_2D(Input, M_upd, Output, sigma, W_halfsize, i1, j1, (long)(dimX), (long)(dimY)); */
    mean_inp2_2D(Input, M_upd, Output, Updated, Gauss_weights, W_halfsize, W_fullsize, sumweigths, i1, j1, (long)(dimX), (long)(dimY));
     }}
    copyIm(Updated, Output, (long)(dimX), (long)(dimY), (long)(dimZ));
    }
    }
    else {
    /* 3D version */
    #pragma omp parallel for shared(Input,M_upd) private(i,j,k)
    for(k=0; k<dimZ; k++) {
      for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
    scaling_func(Input, M_upd, Output, sigma, minmax_array, i, j, k, (long)(dimX), (long)(dimY), (long)(dimZ)); /* scaling function */
    }}}
    for (l=0; l<iterations; l++) {
    #pragma omp parallel for shared(Input,M_upd) private(i1,j1,k1)
    for(k1=0; k1<dimZ; k1++) {
      for(i1=0; i1<dimX; i1++) {
        for(j1=0; j1<dimY; j1++) {
    mean_inp_3D(Input, M_upd, Output, sigma, W_halfsize, i1, j1, k1, (long)(dimX), (long)(dimY),  (long)(dimZ)); /* smoothing of the mask */
    }}}
     }
	   }
    free(Gauss_weights);
    free(minmax_array);
    free(Updated);
    return 0;
    }
}

/********************************************************************/
/**************************COMMON function***************************/
/********************************************************************/
void scaling_func(float *Input, unsigned char *M_upd, float *Output, float sigma, float *minmax_array, long i, long j, long k, long dimX, long dimY, long dimZ)
{
	long  index;
  index = (dimX*dimY)*k + j*dimX+i;

  /* scaling according to the max value in the mask */
  if (M_upd[index] == 1) Output[index] = sigma*(Input[index]/minmax_array[1]);
	return;
}
/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
/*mean smoothing of the inapainted values inside and in the viscinity of the mask */
void mean_inp_2D(float *Input, unsigned char *M_upd, float *Output, float sigma, int W_halfsize, long i, long j, long dimX, long dimY)
{
  long i_m, j_m, i1, j1, index, index2, switcher, counter;
	float sum_val;

  index = j*dimX+i;
  sum_val = 0.0f; switcher = 0; counter = 0;
  for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) {
      i1 = i+i_m;
      if ((i1 < 0) || (i1 >= dimX)) i1 = i;
      for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) {
          j1 = j+j_m;
          if ((j1 < 0) || (j1 >= dimY)) j1 = j;
              index2 = j1*dimX + i1;
              if (M_upd[index2] == 1) switcher = 1;
              sum_val += Output[index2];
              counter++;
      }}
      if (switcher == 1) Output[index] = sum_val/counter;
	return;
}

void mean_inp2_2D(float *Input, unsigned char *M_upd, float *Output, float *Updated, float *Gauss_weights, int W_halfsize, int W_fullsize, float sumweigths, long i, long j, long dimX, long dimY)
{
  long i_m, j_m, i1, j1, z, index, index2, counter, counterglob;
  float *ValVec, multiplier, sum_val;

  ValVec = (float*) calloc(W_fullsize, sizeof(float));

  index = j*dimX+i;
  counter = 0; counterglob = 0; sum_val = 0.0;
  /* check that you're on the region defined by mask */
  if (M_upd[index] == 1) {
  /* check that the pixel to be inpainted is NOT surrounded by zeros of the updated image (Output).
  The idea here is to wait inpainting pixels which are far from the boundaries - to ensure a gradual front progression
  */
  if ((i-1 > 0) && (i+1 < dimX) && (j-1 > 0) && (j+1 < dimY)) {
    if ((Output[j*dimX+(i-1)] != 0.0) || (Output[j*dimX+(i+1)] != 0.0) || (Output[(j-1)*dimX+i] != 0.0) || (Output[(j+1)*dimX+i] != 0.0)) {

      for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) {
          i1 = i+i_m;
          if ((i1 < 0) || (i1 >= dimX)) i1 = i;
          for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) {
              j1 = j+j_m;
              if ((j1 < 0) || (j1 >= dimY)) j1 = j;
                  index2 = j1*dimX + i1;
                  /* */
                  if (Updated[index2] != 0.0) {
                  /*ValVec[counter] = Output[index2]*(Gauss_weights[counterglob]/sumweigths);*/
                  sum_val += Updated[index2];
                  counter++;
                }
                counterglob++;
          }}
      /* if there were non zero mask values */
      if (counter > 0) {
      multiplier = (float)(W_fullsize/counter);
      /*
      for(z=0; z<counter; z++) {
      sum_val += (ValVec[z]/counter)*multiplier;
      }
      */
      Output[index] = sum_val/counter;
      }
    }
  free(ValVec);
	return;
}


/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/
/*mean smoothing of the inapainted values inside and in the viscinity of the mask */
void mean_inp_3D(float *Input, unsigned char *M_upd, float *Output, float sigma, int W_halfsize, long i, long j, long k, long dimX, long dimY, long dimZ)
{
  long i_m, j_m, k_m, i1, j1, k1, index, index2, switcher, counter;
	float sum_val;

  index = (dimX*dimY)*k + j*dimX+i;
  sum_val = 0.0f; switcher = 0; counter = 0;
  for(k_m=-W_halfsize; k_m<=W_halfsize; k_m++) {
    k1 = k+k_m;
    if ((k1 < 0) || (k1 >= dimZ)) k1 = k;
    for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) {
      i1 = i+i_m;
      if ((i1 < 0) || (i1 >= dimX)) i1 = i;
      for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) {
          j1 = j+j_m;
          if ((j1 < 0) || (j1 >= dimY)) j1 = j;

              index2 = (dimX*dimY)*k1 + j1*dimX+i1;
              if (M_upd[index2] == 1) switcher = 1;
              sum_val += Output[index2];
              counter++;
      }}}
      if (switcher == 1) Output[index] = sum_val/counter;
	return;
}
