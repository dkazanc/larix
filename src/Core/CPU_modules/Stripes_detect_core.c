/*
# Copyright 2022 Diamond Light Source Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/

#include "Stripes_detect_core.h"
#include "utils.h"

int StripeWeights_main(float *input, float *output, float *grad_stats, int detectors_window_height, int detectors_window_width, int angles_window_depth, int vertical_mean_window, int gradient_gap, int ncores, long angl_size, long det_X_size, long det_Y_size)
{
    /*
    C module to detect stripes in sinograms (2D) and in projection data (3D). The method involves 3 steps:
    1. Taking first derrivative of the input in the direction orthogonal to stripes.
    2. Slide horizontal rectangular window orthogonal to stripes direction to accenuate outliers (stripes) using median.
    3. Slide the vertical thin (1 pixel) window to calculate a mean (further accenuates stripes).
    
    This method should work for full and partial stripes as well with the changing intensity ones. 
    *
    * Input parameters:
    * 1. sinogram (2D) [angles x detectorsX] OR projection data (3D) [angles x detectorsY x detectorsX]
    * 2. detectors_window_height: (int) the half-height of the searching window parallel to detectors dimension
    * 3. detectors_window_width: (int) the half-width of the searching window parallel to detectors dimension, normally width >> height
    * 4. angles_window_depth: (int) for 3D data, the half-depth of the searching window parallel to angles dimension
    * 5. vertical_mean_window: (int) the half-size of the 1D vertical window to calculate mean inside
    * 6. gradient_gap: (int) the gap in pixels with the neighbour while calculating a gradient (1 is the normal gradient)
    * 7. ncores - number of CPU threads to use (if given), 0 is the default value - using all available cores

    * Output:
    * 1. output - estimated weights (can be thresholded after with Otsu)
    */

    long i, j, k, DimTotal;
    int detectors_window_height_full, detectors_window_width_full, angles_window_depth_full, full_window_size, midval_window_index;
    float *gradientX, *output_temp; 
    
    DimTotal = angl_size*det_X_size*det_Y_size;
    gradientX = calloc(DimTotal, sizeof(float));
    output_temp = calloc(DimTotal, sizeof(float));    
    
    /* define sizes of the searching window */
    detectors_window_height_full = (int)(2*detectors_window_height+1);
    detectors_window_width_full = (int)(2*detectors_window_width+1);
    angles_window_depth_full = (int)(2*angles_window_depth+1); 
    if (det_Y_size == 1) full_window_size = detectors_window_height_full*detectors_window_width_full;
    else full_window_size = detectors_window_height_full*detectors_window_width_full*angles_window_depth_full;
    midval_window_index = (int)(0.5f*full_window_size) - 1;

    if (ncores > 0) {
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(ncores); // Use a number of threads for all consecutive parallel regions 
    }    

    
    /****************************2D INPUT*****************************/ 
    /* calculating the gradient in X direction (det_X_size) */
    if (det_Y_size == 1) gradient2D(input, gradientX, angl_size, det_X_size, 0, gradient_gap); 
    else gradient3D(input, gradientX, angl_size, det_X_size, det_Y_size, 0, gradient_gap); 
    
    /* calculating statistics for gradientX */       
    stats_calc(gradientX, grad_stats, 1, angl_size, det_X_size, det_Y_size);

    //printf("%ld %ld",  angl_size, det_X_size);
    /* We run a rectangular window where detectors_window_width >> detectors_window_height in which we identify the background (median value) and substract 
    it from the current pixel. If there is a stripe (an outlier), it will be accenuated */
    if (det_Y_size == 1) {
    #pragma omp parallel for shared(input, output_temp) private(i,j)
    for(i=0; i<angl_size; i++) {
        for(j=0; j<det_X_size; j++) {
          horiz_median_stride2D(gradientX, output_temp, full_window_size, midval_window_index, detectors_window_height, detectors_window_width, angl_size, det_X_size, i, j);          
        }}

    /* Now we run a 1D vertical window which calculates the mean inside window, i.e. the stripe will be even more prounonced */
    #pragma omp parallel for shared(output, output_temp) private(i,j)
    for(i=0; i<angl_size; i++) {
        for(j=0; j<det_X_size; j++) {
          vert_mean_stride2D(output_temp, output, vertical_mean_window, angl_size, det_X_size, i, j);          
        }}
    }
    else {
    #pragma omp parallel for shared(input, output_temp) private(i,j,k)
    for(i=0; i<angl_size; i++) {
        for(j=0; j<det_X_size; j++) {
            for(k=0; k<det_Y_size; k++) {
          horiz_median_stride3D(gradientX, output_temp, full_window_size, midval_window_index, detectors_window_height, detectors_window_width, angles_window_depth, angl_size, det_X_size, det_Y_size, i, j, k);
        }}}
    #pragma omp parallel for shared(output, output_temp) private(i,j,k)
    for(i=0; i<angl_size; i++) {
        for(j=0; j<det_X_size; j++) {
            for(k=0; k<det_Y_size; k++) {            
          vert_mean_stride3D(output_temp, output, vertical_mean_window, angl_size, det_X_size, det_Y_size, i, j, k);          
        }}}
    }    

    free(gradientX);
    free(output_temp);
    return 0;
}



int StripesMergeMask_main(unsigned char *input, unsigned char *output, int stripe_width_max, int dilate, int ncores, long angl_size, long det_X_size, long det_Y_size)
{
 /*
    C module to merge stripes together into thicker stripes assuming that if there are two stripes in the vicinity of [stripe_width_max] it is probably a single stripe
    *
    * Input parameters:
    * 1. 2D/3D uint8 image/volume: The result of the binary segmentation applied to method (StripeWeights_main). 
    * 2. stripe_width_max: (int) the maximum width of a stripe acceptable
    * 3. dilate (int) : the number of pixels/voxels to dilate obtained mask
    * 4. ncores - number of CPU threads to use (if given), 0 is the default value - using all available cores

    * Output:
    * 1. output - merged mask with full stripes
    */
    long i, j, k, DimTotal;
    unsigned char *output_temp; 
    
    DimTotal = angl_size*det_X_size*det_Y_size;
    output_temp = calloc(DimTotal, sizeof(unsigned char));    
   
    /* copying input to output */
    copyIm_unchar(input, output, angl_size, det_X_size, det_Y_size);
    
    if (ncores > 0) {
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(ncores); // Use a number of threads for all consecutive parallel regions 
    }    
    
    if (det_Y_size == 1) {
    /* 2D version */
    #pragma omp parallel for shared(input, output) private(i,j)
    for(i=0; i<angl_size; i++) {
        for(j=0; j<det_X_size; j++) {
          stripes_merger2D(input, output, stripe_width_max, angl_size, det_X_size, i, j);          
    }}

    if (dilate > 0){
        /* perform mask dilation */
        for(k=0; k<dilate; k++) {
        copyIm_unchar(output, output_temp, angl_size, det_X_size, det_Y_size);
        mask_dilate2D(output, output_temp, angl_size, det_X_size);
        copyIm_unchar(output_temp, output, angl_size, det_X_size, det_Y_size);
            }
        }
    }
    else {
    /*3D version (just a 2D extended) */

    }
    free(output_temp);
    return 0;
}

/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
void horiz_median_stride2D(float *input, float *output, int full_window_size, int midval_window_index, int detectors_window_height, int detectors_window_width, long angl_size, long det_X_size, long i, long j)
{
    long index; 
    float *_values;
    _values = (float*) calloc(full_window_size, sizeof(float));
    
    index = j*angl_size+i;
  
    /* get the values into the searching window allocated vector */
    fill_vector_with_neigbours2D(input, _values, detectors_window_height, detectors_window_width, angl_size, det_X_size, i, j);
    quicksort_float(_values, 0, full_window_size-1); 
    output[index] = fabs(input[index]) - fabs(_values[midval_window_index]);

    free(_values);
}

void vert_mean_stride2D(float *input, float *output, int vertical1D_halfsize, long angl_size, long det_X_size, long i, long j)
{
    long j_m, j1, counter_local, index, index2;
    float sumval;

    index = j*angl_size+i;   
    
    /* calculates mean along 1D vertical kernel */
    counter_local = 0; sumval = 0.0f;
    for(j_m=-vertical1D_halfsize; j_m<=vertical1D_halfsize; j_m++) {
        j1 = j+j_m;
            if ((j1 >= 0) && (j1 < det_X_size)) {
                index2 = j1*angl_size + i;
                if (input[index2] != 0.0f){
                sumval += input[index2];
                counter_local++;
                }                
            }
        }

    if (counter_local != 0) output[index] = sumval/counter_local;
}

void stripes_merger2D(unsigned char *input, unsigned char *output, int stripe_width_max, long angl_size, long det_X_size, long i, long j)
{
    /* the function that checks if two stripes are close to each other based on stripe_width_max and merge them together */    
    long i_m, i1, i_m2, i2, index, index2; 
    index = j*angl_size+i; 

    if (input[index] == 1) {
    /* one edge of the mask has found, lets look for another one */

    for(i_m=stripe_width_max; i_m>=0; i_m--) {
        i1 = i+i_m;
            if ((i1 >= 0) && (i1 < angl_size)) {                      
                index2 = j*angl_size + i1; 
                if (input[index2] == 1) {
                /* second edge of the mask has found, convert all values in-between to ones */
                for(i_m2 = 0; i_m2 < i_m; i_m2++) {
                i2 = i + i_m2;
                output[j*angl_size + i2] = 1;
                    }
                break;
                }
            }
        }
    }    
}


/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/

void horiz_median_stride3D(float *input, float *output, int full_window_size, int midval_window_index, int detectors_window_height, int detectors_window_width, int angles_window_depth, long angl_size, long det_X_size, long det_Y_size, long i, long j, long k)
{
    long index; 
    float *_values;
    _values = (float*) calloc(full_window_size, sizeof(float));
    
    index = (angl_size*det_X_size)*k + j*angl_size+i;

    /* get the values into the searching window allocated vector */
    fill_vector_with_neigbours3D(input, _values, detectors_window_height, detectors_window_width, angles_window_depth, angl_size, det_X_size, det_Y_size, i, j, k);
    quicksort_float(_values, 0, full_window_size-1); 
    output[index] = fabs(input[index]) - fabs(_values[midval_window_index]);    
    free(_values);
}

void vert_mean_stride3D(float *input, float *output, int vertical1D_halfsize, long angl_size, long det_X_size, long det_Y_size, long i, long j, long k)
{
    long k_m, k1, counter_local, index, index2;
    float sumval;

    index = (angl_size*det_X_size)*k + j*angl_size+i;  
    
    /* calculates mean along 1D vertical kernel */    
    counter_local = 0; sumval = 0.0f;
    for(k_m=-vertical1D_halfsize; k_m<=vertical1D_halfsize; k_m++) {
        k1 = k+k_m;
            if ((k1 >= 0) && (k1 < det_Y_size)) {
                index2 = (angl_size*det_X_size)*k1 + j*angl_size + i;
                if (input[index2] != 0.0f){
                sumval += input[index2];
                counter_local++;
                }                
            }
        }

    if (counter_local != 0) output[index] = sumval/counter_local;    

}