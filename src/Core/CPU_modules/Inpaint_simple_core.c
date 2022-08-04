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

/* Simple morphological inpainting schemes which are progressing from the edge inwards, 
 * therefore acting like a diffusion-type process 
 *
 * Input Parameters:
 * 1. Image/volume to inpaint
 * 2. Mask of the same size as (1) in 'unsigned char' format  (ones mark the region to inpaint, zeros belong to the data)
 * 3. Iterations number
 * 4. Half-window size of the searching window
 * 5. method type to select an inpainting value: 0 - mean, 1 - meadian, 2 - random neighbour
 *
 * Output:
 * [1] Inpainted image
 */

int Inpaint_simple_CPU_main(float *Input, unsigned char *Mask, float *Output, unsigned char *M_upd, int iterations, int W_halfsize, int method_type, int dimX, int dimY, int dimZ)
{
    long i, j, k, l, countmask, DimTotal, iterations_mask_complete;
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
    if ((method_type == 1) || (method_type == 2)) median_rand_inpainting_2D(Input, M_upd, Output, Updated, W_halfsize, W_fullsize, method_type, i, j, (long)(dimX), (long)(dimY));
    else eucl_weighting_inpainting_2D(Input, M_upd, Output, Updated, Gauss_weights, W_halfsize, W_fullsize, i, j, (long)(dimX), (long)(dimY));
    }}
    copyIm(Updated, Output, (long)(dimX), (long)(dimY), (long)(dimZ));

    /* check here if the iterations to complete the masked region needs to be terminated */
    countmask = 0;
    for (k=0; k<DimTotal; k++) {
      if (M_upd[k] == 1) countmask++;
    }
    if (countmask == 0) {
      break; /*exit iterations_mask_complete loop */
      }
    }
    /* performing user defined iterations */
        for (l=0; l<iterations; l++) {
    #pragma omp parallel for shared(Input,M_upd,Gauss_weights) private(i,j)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
    eucl_weighting_inpainting_2D(Input, M_upd, Output, Updated, Gauss_weights, W_halfsize, W_fullsize, i, j, (long)(dimX), (long)(dimY));}}    
    copyIm(Updated, Output, (long)(dimX), (long)(dimY), (long)(dimZ));
    copyIm_unchar(Mask, M_upd, dimX, dimY, dimZ);    
        }
    }
    else {
    /* 3D version */
	   }
    free(Gauss_weights);
    free(Updated);
    return 0;
}

/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
void eucl_weighting_inpainting_2D(float *Input, unsigned char *M_upd, float *Output, float *Updated, float *Gauss_weights, int W_halfsize, int W_fullsize, long i, long j, long dimX, long dimY)
{
  /* applying inpainting with euclidian (distance) weighting */
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

void median_rand_inpainting_2D(float *Input, unsigned char *M_upd, float *Output, float *Updated, int W_halfsize, int W_fullsize, int method_type, long i, long j, long dimX, long dimY)
{
  long i_m, j_m, i1, j1, index, index2;
  float vicinity_mean;
  float *_values;
  int counter_local, r, median_val;
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

  /* fill the vectors */
  counter_local = 0;
      for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) {
          i1 = i+i_m;
          for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) {
              j1 = j+j_m;
              if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
                  index2 = j1*dimX + i1;
                  if (Output[index2] != 0.0) {
                  _values[counter_local] = Output[index2];
                  counter_local++;
                  }
              }
          }}   
  if (method_type == 1) {
  /* inpainting based on the median neighbour */
  sort_quick(_values, 0, counter_local); 
  median_val = (int)(counter_local/2);
  Updated[index] = _values[median_val];
  }
  else {
  /* inpainting based on a random neighbour */
  r = rand() % counter_local;
  Updated[index] = _values[r];
  }
  M_upd[index] = 0;
      }
  }  
  free(_values);
	return;
}

/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/