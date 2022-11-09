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

/*
* C module to detect stripes in sinograms (2D) and projection data (3D).
The input could be a normal image or a gradient in orthogonal to stripes direction. 

The sliding window orthogonal to stripes in which we calculate a measure which detects outlliers, e.g. median or
Median absolute deviation. 

The method should work for full and partial stripes as well with the changing intensity ones
*
* Input parameters:
* 1. sinogram (2D) [angles x detectorsX] OR projection data (3D) [detectorsX x angles x detectorsY]
* 2. detectors_window_height: (int) the half-height of the searching window parallel to detectors dimension
* 3. detectors_window_width: (int) the half-width of the searching window parallel to detectors dimension, normally width >> height
* 4. angles_window_depth: (int) for 3D data, the half-depth of the searching window parallel to angles dimension
* 5. gradient_calc: (int) 1 - take a gradient (in detectorsX dim) of the input image, 0 - do not calculate
* 6. ncores - number of CPU threads to use (if given), 0 is the default value - using all available cores

* Output:
* 1. output - estimated weights
*/

int StripeWeights_main(float *input, float *output, int detectors_window_height, int detectors_window_width, int angles_window_depth, int gradient_calc, int ncores, long angl_size, long det_X_size, long det_Y_size)
{
    long i, j, k, DimTotal;
    int detectors_window_height_full, detectors_window_width_full, angles_window_depth_full, full_window_size, midval_window_index, vertical1D_halfsize;
    float *gradientX, *output_temp; 
    vertical1D_halfsize = 5;
    
    DimTotal = angl_size*det_X_size*det_Y_size;
    gradientX = calloc(1, sizeof(float));
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

    if (det_Y_size == 1) {
    /****************************2D INPUT*****************************/    
    if (gradient_calc == 1) {
    free(gradientX);
    gradientX = calloc(DimTotal, sizeof(float));
    /* calculating the gradient in X direction (det_X_size) */        
    gradient2D(input, gradientX, angl_size, det_X_size, 0);        
    }

    //printf("%ld %ld",  angl_size, det_X_size);
    /* We run a rectangular window where detectors_window_width >> detectors_window_height in which we identify the background (median value) and substract 
    it from the current pixel. If there is a stripe (an outlier) it will be accenuated */
    #pragma omp parallel for shared(input, output_temp) private(i,j)
    for(i=0; i<angl_size; i++) {
        for(j=0; j<det_X_size; j++) {
          if (gradient_calc == 1) horiz_median_stride2D(gradientX, output_temp, full_window_size, midval_window_index, detectors_window_height, detectors_window_width, angl_size, det_X_size, i, j);
          else horiz_median_stride2D(input, output_temp, full_window_size, midval_window_index, detectors_window_height, detectors_window_width, angl_size, det_X_size, i, j);
    }}
    /* Now we run a 1D vertical window which calculates the mean inside window, i.e. the stripe will be even more prounonced */
    #pragma omp parallel for shared(output, output_temp) private(i,j)
    for(i=0; i<angl_size; i++) {
        for(j=0; j<det_X_size; j++) {
          vert_mean_stride2D(output_temp, output, vertical1D_halfsize, angl_size, det_X_size, i, j);          
    }}

    }
    else {

    }   
    free(gradientX);
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