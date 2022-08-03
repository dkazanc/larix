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

int Inpaint_simple_CPU_main(float *Input, unsigned char *Mask, float *Output, unsigned char *M_upd, int iterations, int W_halfsize, int dimX, int dimY, int dimZ)
{
    long i, j, k, i1, j1, k1, l, countmask, DimTotal, iterations_mask_complete;
    int i_m, j_m;
    float *Gauss_weights, *Updated=NULL;
    int W_fullsize, counter;

    DimTotal = (long)(dimX*dimY*dimZ);
    Updated = calloc(DimTotal, sizeof(float));

    /* copy input into output */
    copyIm(Input, Output, (long)(dimX), (long)(dimY), (long)(dimZ));
    copyIm(Input, Updated, (long)(dimX), (long)(dimY), (long)(dimZ));
    /* copying M to Mask_upd */
    copyIm_unchar(Mask, M_upd, dimX, dimY, dimZ);

    /* pre-calculation of Gaussian distance weights  */
    W_fullsize = (int)(2*W_halfsize + 1); /*full size of similarity window */
    Gauss_weights = (float*)calloc(W_fullsize*W_fullsize,sizeof(float ));
    counter = 0;
    for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) {
        for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) {
            Gauss_weights[counter] = expf(-(powf((i_m), 2) + powf((j_m), 2))/(2*W_halfsize*W_halfsize));
            counter++;
        }
    }
    /*calculate all nonzero values in the given mask */
    countmask = 0;
    for (k=0; k<DimTotal; k++) {
      if (Mask[k] == 1) countmask++;
    }
    if (countmask == 0) {
      free(Updated);
      free(Gauss_weights);
      return 0;
    }
    iterations_mask_complete = countmask; /*the maximum number of required iterations to do the completion of the inpainted region */

    if (dimZ == 1) {
    for (l=0; l<iterations_mask_complete; l++) {
    #pragma omp parallel for shared(Input,M_upd,Gauss_weights) private(i,j)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
    //mean_inpainting_2D(Input, M_upd, Output, Updated, Gauss_weights, W_halfsize, W_fullsize, i, j, (long)(dimX), (long)(dimY));
    patch_selective_inpainting_2D(Input, M_upd, Output, Updated, W_halfsize, W_fullsize, i, j, (long)(dimX), (long)(dimY));
    }}
    copyIm(Updated, Output, (long)(dimX), (long)(dimY), (long)(dimZ));

    /*check here if the iterations to complete the masked region needs to be terminated */
    countmask = 0;
    for (k=0; k<DimTotal; k++) {
      if (M_upd[k] == 1) countmask++;
    }
    if (countmask == 0) {
      break; /*exit iterations_mask_complete loop */
      }
    }
    /*performing user defined iterations */
        for (l=0; l<iterations; l++) {
    #pragma omp parallel for shared(Input,M_upd,Gauss_weights) private(i,j)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
    mean_inpainting_2D(Input, M_upd, Output, Updated, Gauss_weights, W_halfsize, W_fullsize, i, j, (long)(dimX), (long)(dimY));}}
    //patch_selective_inpainting_2D(Input, M_upd, Output, Updated, W_halfsize, W_fullsize, i, j, (long)(dimX), (long)(dimY));}}        
    copyIm(Updated, Output, (long)(dimX), (long)(dimY), (long)(dimZ));
    copyIm_unchar(Mask, M_upd, dimX, dimY, dimZ);    
        }
    }
    else {
    /* 3D version */
    /*
    #pragma omp parallel for shared(Input,M_upd) private(i,j,k)
    for(k=0; k<dimZ; k++) {
      for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
    scaling_func(Input, M_upd, Output, sigma, minmax_array, i, j, k, (long)(dimX), (long)(dimY), (long)(dimZ));
    }}}
    for (l=0; l<iterations; l++) {
    #pragma omp parallel for shared(Input,M_upd) private(i1,j1,k1)
    for(k1=0; k1<dimZ; k1++) {
      for(i1=0; i1<dimX; i1++) {
        for(j1=0; j1<dimY; j1++) {
    mean_inp_3D(Input, M_upd, Output, sigma, W_halfsize, i1, j1, k1, (long)(dimX), (long)(dimY),  (long)(dimZ));
    }}}
     }
     */
	   }
    free(Gauss_weights);
    free(Updated);
    return 0;
}

/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
void mean_inpainting_2D(float *Input, unsigned char *M_upd, float *Output, float *Updated, float *Gauss_weights, int W_halfsize, int W_fullsize, long i, long j, long dimX, long dimY)
{
  long i_m, j_m, i1, j1, index, index2;
  float sum_val, sumweights;
  int counter_local, counterglob, counter_vicinity;
  index = j*dimX+i;
  /* check that you're on the region defined by the updated mask */
  if (M_upd[index] == 1) {
    /*check if have a usable information in the vicinity of the mask's edge*/
  counter_vicinity = 0;
  for(i_m=-1; i_m<=1; i_m++) {
      i1 = i+i_m;
      for(j_m=-1; j_m<=1; j_m++) {
          j1 = j+j_m;
          if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
            if (Output[j1*dimX+i1] != 0.0){
            counter_vicinity++;
            break;
            }
    }
  }}
  if (counter_vicinity > 0) {

      counter_local = 0; sum_val = 0.0f; sumweights = 0.0f; counterglob = 0;
      for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) {
          i1 = i+i_m;
          for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) {
              j1 = j+j_m;
              if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
                  index2 = j1*dimX + i1;
                  if (Output[index2] != 0.0) {
                  sum_val += Output[index2]*Gauss_weights[counterglob];
                  sumweights += Gauss_weights[counterglob];
                  counter_local++;
                }
              }
            counterglob++;
          }}
      /* if there were non zero mask values */
      if (counter_local > 0) {
      Updated[index] = sum_val/sumweights;
      M_upd[index] = 0;
      }
        }

    }
	return;
}

void patch_selective_inpainting_2D(float *Input, unsigned char *M_upd, float *Output, float *Updated, int W_halfsize, int W_fullsize, long i, long j, long dimX, long dimY)
{
  long i_m, j_m, i1, j1, k, index, index2;
  float vicinity_mean, sumweight;
  float *_differences, *_values;
  int counter_local, neighbors_add, r;
  _differences = (float*) calloc(W_fullsize*W_fullsize, sizeof(float));
  _values = (float*) calloc(W_fullsize*W_fullsize, sizeof(float));

  index = j*dimX+i;
  /* check that you're on the region defined by the updated mask */
  if (M_upd[index] == 1) {
  /*check if have a usable information in the vicinity of the mask's edge*/
  counter_local = 0; vicinity_mean = 0.0f;
  for(i_m=-1; i_m<=1; i_m++) {
      i1 = i+i_m;
      for(j_m=-1; j_m<=1; j_m++) {
          j1 = j+j_m;
          if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
            if (Output[j1*dimX+i1] != 0.0){
            vicinity_mean += Output[j1*dimX+i1];
            counter_local++;
            }
    }
  }}
  /*If we've got usable data in the vicinity then proceed with inpainting */
  if (vicinity_mean != 0.0f) {
  vicinity_mean = vicinity_mean/counter_local; /* get the mean of values in the vicinity */

  sumweight = 0.0f;
  /* fill the vectors */
  counter_local = 0;
      for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) {
          i1 = i+i_m;
          for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) {
              j1 = j+j_m;
              if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
                  index2 = j1*dimX + i1;
                  if (Output[index2] != 0.0) {
                  _differences[counter_local] = fabs(Output[index2]-vicinity_mean);
                  _values[counter_local] = Output[index2];
                  counter_local++;
                  }
              }
          }}

  /* performing sorting of values in vectors according to the abs difference */
    // sort_quick(_differences, 0, counter_local); 
    
  r = rand() % counter_local;

  /*
    neighbors_add = ceil((float)counter_local/2);    
    for(k=0; k<neighbors_add; k++) {
      sumweight += _differences[0]+vicinity_mean;
    }
    */

    Updated[index] = _values[r];
    M_upd[index] = 0;
      }
  }
  free(_differences);
  free(_values);
	return;
}

/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/
/*mean smoothing of the inapainted values inside and in the viscinity of the mask */
void mean_inp_3D(float *Input, unsigned char *M_upd, float *Output, int W_halfsize, long i, long j, long k, long dimX, long dimY, long dimZ)
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
