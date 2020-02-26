/* This works has been developed at Diamond Light Source Ltd.
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

#include "autocropper_core.h"
#include "utils.h"

/* A data cropping algorithm where the object of interest lies within the FOV. 
 * The algorithm is developed mainly for cropping of tomographic projection data
 *
 * Input Parameters (from Python):
 * Output: 
 */


float Autocrop_main(float *Input, float *mask_box, float *crop_indeces, float threshold, int margin_skip, int statbox_size, int increase_crop, int dimX, int dimY, int dimZ)
{
    //long i;    
    float *maskMean_value;
    maskMean_value = (float*) calloc (1,sizeof(float));
        
    if (dimZ == 1) {
    /* 2D processing */

    /* collecting statistics in the box */        
    stat_collector2D(Input, maskMean_value, statbox_size, dimX, dimY);    
    //printf("%f\n", maskMean_value[0]);       

    /* calculating the difference of a mean of 3x3 pixel with the mean obtained in the mask */
    diff_mask2D(Input, mask_box, maskMean_value, margin_skip, dimX, dimY);        
    
    /* getting the indeces to crop */
    get_indices2D(Input, mask_box, crop_indeces, threshold, increase_crop, dimX, dimY);        
    
    }
    else {
    }

    free(maskMean_value);
    return *crop_indeces; 
}

/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
float stat_collector2D(float *Input, float *maskMean_value, int statbox_size, int dimX, int dimY)
{
    /* the module places a box in the background region of the data in order to collect statistics */
    int statbox_size_vert, mid_vert_index_Y, vertBox_index_up, vertBox_index_down;
    long i, j, counter;
    float meanval;

    mid_vert_index_Y = (int)(0.5f*dimY);
    statbox_size_vert = (int)(2.5f*statbox_size);
    /* place the box in the vertical middle of the image */
    vertBox_index_up = mid_vert_index_Y-statbox_size_vert;
    vertBox_index_down = mid_vert_index_Y+statbox_size_vert;    
 
    /* collecting statistics in the box */
    counter = 0; meanval = 0.0f;
    for(j=0; j<dimY; j++) {
        if ((j >= vertBox_index_up) && (j <= vertBox_index_down)) {
        for(i=0; i<dimX; i++) {        
            if (((i >= 0) && (i <= statbox_size)) || ((i >= dimX-statbox_size) && (i < dimX))) {
            meanval += Input[j*dimX+i];
            counter++;
	            }
        	}
	   }
         }
    maskMean_value[0] = meanval/(float)(counter);    
    return *maskMean_value;
}

float diff_mask2D(float *Input, float *mask_box, float *maskMean_value, int margin_skip, int dimX, int dimY)
{ 	 
    long i, j, i1, j1, j_m, i_m;
    float local_mean;

#pragma omp parallel for shared (Input, mask_box, maskMean_value) private(i, j, i1, j1, j_m, i_m, local_mean)
    for(j=0; j<dimY; j++) {
     if ((j > margin_skip) && (j < dimY-margin_skip)) {
        for(i=0; i<dimX; i++) {        
            if ((i > margin_skip) && (i < dimX-margin_skip)) {
               local_mean = 0.0f;
               for(j_m=-1; j_m<=1; j_m++) {
                   for(i_m=-1; i_m<=1; i_m++) {
                    i1 = i+i_m;
                    j1 = j+j_m;
                    local_mean += Input[j1*dimX+i1];
	            }}
		    local_mean /= 9.0f;
		    mask_box[j*dimX+i] = fabs(local_mean - maskMean_value[0]);
                } 
              }           
           }
         }
      return *mask_box;
}

float get_indices2D(float *Input, float *mask_box, float *crop_indeces, float threshold, int increase_crop, int dimX, int dimY)
{	
	float *MeanX_vector, *MeanY_vector;
	MeanX_vector = (float*) calloc (dimX,sizeof(float));
	MeanY_vector = (float*) calloc (dimY,sizeof(float));

	long i, j;
	float maxval, meanval;
	
	/* get the maximum value inside the differences mask */
	maxval = 0.0f;
	for(j=0; j<dimY*dimX; j++) {
	if (mask_box[j] > maxval) {
         	maxval = mask_box[j];
	  }
	}
	
	/*get X-dim mean vector*/
	for(i=0; i<dimX; i++) {        
	  meanval = 0.0;
	  for(j=0; j<dimY; j++) {
	  meanval += mask_box[j*dimX+i];
	  }
	MeanX_vector[i] = meanval/dimY;
	}	
	
	/*get Y-dim mean vector*/	
	for(j=0; j<dimY; j++) {
	  meanval = 0.0;
          for(i=0; i<dimX; i++) {        
          meanval += mask_box[j*dimX+i];
          }
	MeanY_vector[j] = meanval/dimX;
	}
	
	/* find the first index of X-dim */
	for(i=0; i<dimX; i++) {
	if (MeanX_vector[i] >= threshold*maxval) {
	crop_indeces[0] = i - increase_crop; /* first X-index*/
	break;	
           }
           else crop_indeces[0] = 0; /* first X-index*/
	}
	if (crop_indeces[0] < 0) crop_indeces[0] = 0;
	
	for(i=dimX-1; i>=0; i--) {
	if (MeanX_vector[i] >= threshold*maxval) {
	crop_indeces[1] = i + increase_crop; /* second X-index*/
	break;	
        }
        else crop_indeces[1] = dimX; /* second X-index*/
        }
        
      	if (crop_indeces[1] > dimX) crop_indeces[1] = dimX;
	
	/* find the first index of Y-dim */
	for(j=0; j<dimY; j++) {
	if (MeanY_vector[j] >= threshold*maxval) {
	crop_indeces[2] = j-increase_crop; /* first Y-index*/
	break;	
           }
        else crop_indeces[2] = 0; /* first Y-index*/
	}		
	if (crop_indeces[2] < 0) crop_indeces[2] = 0;
	
	for(j=dimY-1; j>=0; j--) {
	if (MeanY_vector[j] >= threshold*maxval) {
	crop_indeces[3] = j+increase_crop; /* second Y-index*/
	break;	
         }
        else crop_indeces[3] = dimY; /* second Y-index*/
	}	
	if (crop_indeces[3] > dimY) crop_indeces[3] = dimY;
	
	free(MeanX_vector);
 	free(MeanY_vector);
	
        return *crop_indeces;			
}
