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

/********************************************************************/
/*************************stripesdetect3d****************************/
/********************************************************************/
int stripesdetect3d_main_float(float* Input, float* Output, 
                           int window_halflength_vertical,
                           int ratio_radius,
                           int ncores,
                           long dimX, long dimY, long dimZ)
{
    long      i;
    long      j;
    long      k;
    long long totalvoxels;    
    totalvoxels = (long long)(dimX*dimY*dimZ);

    int window_fulllength = (int)(2*window_halflength_vertical + 1);
    int midval_window_index = (int)(0.5f*window_fulllength) - 1;
    
    float* temp3d_arr;
    temp3d_arr = malloc(totalvoxels * sizeof(float));
    if (temp3d_arr == NULL) printf("Allocation of the 'temp3d_arr' array failed");
    
    /* dealing here with a custom given number of cpu threads */
    if(ncores > 0)
    {
        // Explicitly disable dynamic teams
        omp_set_dynamic(0);
        // Use a number of threads for all consecutive parallel regions
        omp_set_num_threads(ncores);
    }

/* Perform a gentle (6-stencil) 3d mean smoothing of the data to ensure more stability in the gradient calculation */
#pragma omp parallel for shared(temp3d_arr) private(i, j, k)
        for(k = 0; k < dimZ; k++)
        {
            for(j = 0; j < dimY; j++)
            {
                for(i = 0; i < dimX; i++)
                {
                    mean_stride3d(Input, temp3d_arr,
                                  i, j, k, 
                                  dimX, dimY, dimZ);
                }
            }
        }

    /* Take the gradient in the horizontal direction, axis = 0, step = 2*/
    gradient3D_local(Input, Output, dimX, dimY, dimZ, 0, 2);
    
    /*
    Here we calculate a ratio between the mean in a small 2D neighbourhood parallel to the stripe 
    and the mean orthogonal to the stripe. The gradient variation in the direction orthogonal to the
    stripe is expected to be large (a jump), while in parallel direction small. Therefore at the edges
    of a stripe we should get a ratio small/large or large/small. 
    */
#pragma omp parallel for shared(Output, temp3d_arr) private(i, j, k)
        for(k = 0; k < dimZ; k++)
        {
            for(j = 0; j < dimY; j++)
            {
                for(i = 0; i < dimX; i++)
                {
                    ratio_mean_stride3d(Output, temp3d_arr, 
                                        ratio_radius, 
                                        i, j, k,
                                        dimX, dimY, dimZ);
                }
            }
        }
        
    /* 
    We process the resulting ratio map with a vertical median filter which removes 
    inconsistent from longer stripes features
    */
#pragma omp parallel for shared(temp3d_arr, Output) private(i, j, k)
        for(k = 0; k < dimZ; k++)
        {
            for(j = 0; j < dimY; j++)
            {
                for(i = 0; i < dimX; i++)
                {
                    vertical_median_stride3d(temp3d_arr, Output, 
                                             window_halflength_vertical, 
                                             window_fulllength,
                                             midval_window_index,
                                             i, j, k, 
                                             dimX, dimY, dimZ);
                }
            }
        }

    free(temp3d_arr);
    return 0;
}


/********************************************************************/
/*************************stripesmask3d******************************/
/********************************************************************/
int stripesmask3d_main_float(float* Input,
                             unsigned char* Output,
                             float threshold_val,
                             int stripe_length_min,
                             int stripe_depth_min,
                             int stripe_width_min,
                             float sensitivity,
                             int ncores, long dimX, long dimY, long dimZ)
{
    long      i;
    long      j;
    long      k;
    size_t index;
    long long totalvoxels;
    totalvoxels = (long long)(dimX*dimY*dimZ);

    unsigned char* mask;    
    mask = malloc(totalvoxels * sizeof(unsigned char));
    if (mask == NULL) printf("Allocation of the 'mask' array failed");

    /* dealing here with a custom given number of cpu threads */
    if(ncores > 0)
    {
        // Explicitly disable dynamic teams
        omp_set_dynamic(0);
        // Use a number of threads for all consecutive parallel regions
        omp_set_num_threads(ncores);
    }

    /* 
    First step is to mask all the values in the given weights input image    
    that are bellow a given "threshold_val" parameter 
    */
#pragma omp parallel for shared(Input, mask) private(i, j, k, index)
        for(k = 0; k < dimZ; k++)
        {
            for(j = 0; j < dimY; j++)
            {
                for(i = 0; i < dimX; i++)
                {
                    index = (size_t)(dimX * dimY * k) + (size_t)(j * dimX + i);
                    if (Input[index] <= threshold_val) 
                    {
                        mask[index] = 1;
                    }
                    
                }
            }
        }
    /* 
    Now we need to remove stripes that are shorter than "stripe_length_min" parameter
    or inconsistent otherwise. For every pixel we will run a 1D vertical window to count 
    nonzero values in the mask. We also check for the depth of the mask's value, 
    assuming that the stripes are normally shorter in depth compare to the features that 
    belong to true data.
    */
#pragma omp parallel for shared(mask, Output) private(i, j, k)
        for(k = 0; k < dimZ; k++)
        {
            for(j = 0; j < dimY; j++)
            {
                for(i = 0; i < dimX; i++)
                {
                    
                    remove_inconsistent_stripes(mask, Output,
                                                stripe_length_min,
                                                stripe_depth_min,
                                                sensitivity,
                                                i, j, k,
                                                dimX, dimY, dimZ);
                }
            }
        }
    /* Copy output to mask */
   copyIm_unchar(Output, mask, dimX, dimY, dimZ);

    /* We can merge stripes together if they are relatively close to each other
     based on the stripe_width_min parameter */
#pragma omp parallel for shared(mask, Output) private(i, j, k)
        for(k = 0; k < dimZ; k++)
        {
            for(j = 0; j < dimY; j++)
            {
                for(i = 0; i < dimX; i++)
                {
                    merge_stripes(mask, Output,
                                  stripe_width_min,
                                  i, j, k, 
                                  dimX, dimY, dimZ);
                }
            }
        }

    free(mask);
    return 0;
}
/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/
/* Calculate the forward difference derrivative of the 3D input in the direction of the "axis" parameter 
using the step_size in pixels to skip pixels (i.e. step_size = 1 is the classical gradient)
axis = 0: horizontal direction
axis = 1: depth direction
axis = 2: vertical direction
*/
void 
gradient3D_local(float *input, float *output, long dimX, long dimY, long dimZ, int axis, int step_size)
{  
    long i;
    long j;
    long k;
    long i1;
    long j1;
    long k1;
    size_t index;
   
#pragma omp parallel for shared(input, output) private(i,j,k,i1,j1,k1,index)
    for(j=0; j<dimY; j++)     
    {
        for(i=0; i<dimX; i++) 
        {
            for(k=0; k<dimZ; k++)             
            {
            index = (size_t)(dimX * dimY * k) + (size_t)(j * dimX + i);
                /* Forward differences */
                if (axis == 0) 
                {
                    i1 = i + step_size; 
                    if (i1 >= dimX) 
                        i1 = i - step_size;
                    output[index] = input[(size_t)(dimX * dimY * k) + (size_t)(j * dimX + i1)] - input[index];
                }
                else if (axis == 1) 
                {
                    j1 = j + step_size; 
                    if (j1 >= dimY) 
                        j1 = j - step_size;
                    output[index] = input[(size_t)(dimX * dimY * k) + (size_t)(j1 * dimX + i)] - input[index];
                }
                else 
                {
                    k1 = k + step_size; 
                    if (k1 >= dimZ) 
                        k1 = k-step_size;
                    output[index] = input[(size_t)(dimX * dimY * k1) + (size_t)(j * dimX + i)] - input[index];
                }
            }
        }
    }
}

void
ratio_mean_stride3d(float* input, float* output,
                    int radius,
                    long i, long j, long k, 
                    long dimX, long dimY, long dimZ)
{
    float mean_plate;
    float mean_horiz;
    float mean_horiz2;
    float min_val;    
    int diameter = 2*radius + 1;
    int all_pixels_window = diameter*diameter;
    long      i_m;
    long      j_m;
    long      k_m;
    long      i1;
    long      j1;
    long      k1;
    size_t      index;
    size_t      newindex;
    
    index = (size_t)(dimX * dimY * k) + (size_t)(j * dimX + i);

    /* calculate mean of gradientX in a 2D plate parallel to stripes direction */
    mean_plate = 0.0f;
    for(j_m = -radius; j_m <= radius; j_m++)
    {
        j1 = j + j_m;
        if ((j1 < 0) || (j1 >= dimY))
            j1 = j - j_m;
        for(k_m = -radius; k_m <= radius; k_m++)
        {
            k1 = k + k_m;
            if((k1 < 0) || (k1 >= dimZ))
               k1 = k - k_m;
            newindex = (size_t)(dimX * dimY * k1) + (size_t)(j1 * dimX + i);
            mean_plate += fabsf(input[newindex]);
        }
    }
    mean_plate /= (float)(all_pixels_window);
    output[index] = mean_plate;
    
    /* calculate mean of gradientX in a 2D plate orthogonal to stripes direction */
    mean_horiz = 0.0f;
    for(j_m = -1; j_m <= 1; j_m++)
    {
        j1 = j + j_m;
        if((j1 < 0) || (j1 >= dimY))
            j1 = j - j_m;
        for(i_m = 1; i_m <= radius; i_m++)
        {
            i1 = i + i_m;
            if (i1 >= dimX) 
                i1 = i - i_m;
            newindex = (size_t)(dimX * dimY * k) + (size_t)(j1 * dimX + i1);
            mean_horiz += fabsf(input[newindex]);
        }
    }
    mean_horiz /= (float)(radius*3);
    
    /* Calculate another mean symmetrically */
    mean_horiz2 = 0.0f;
    for(j_m = -1; j_m <= 1; j_m++)
    {
        j1 = j + j_m;
        if((j1 < 0) || (j1 >= dimY))
            j1 = j - j_m;
        for(i_m = -radius; i_m <= -1; i_m++)
        {
            i1 = i + i_m;
            if (i1 < 0)
                i1 = i - i_m;
            newindex = (size_t)(dimX * dimY * k) + (size_t)(j1 * dimX + i1);
            mean_horiz2 += fabsf(input[newindex]);
        }
    }
    mean_horiz2 /= (float)(radius*3);

    /* calculate the ratio between two means assuming that the mean 
    orthogonal to stripes direction should be larger than the mean 
    parallel to it */
    if ((mean_horiz > mean_plate) && (mean_horiz != 0.0f))
    {        
        output[index] = mean_plate/mean_horiz;
    }
    if ((mean_horiz < mean_plate) && (mean_plate != 0.0f))
    {        
        output[index] = mean_horiz/mean_plate;
    }    
    min_val = 0.0f;
    if ((mean_horiz2 > mean_plate) && (mean_horiz2 != 0.0f))
    {   
        min_val = mean_plate/mean_horiz2;
    }
    if ((mean_horiz2 < mean_plate) && (mean_plate != 0.0f))
    {
        min_val = mean_horiz2/mean_plate;
    }

    /* accepting the smallest value */
    if (output[index] > min_val)
    {
        output[index] = min_val;
    }

    return;
}

void
vertical_median_stride3d(float* input, float* output,
                        int window_halflength_vertical,
                        int window_fulllength,
                        int midval_window_index,
                        long i, long j, long k,
                        long dimX, long dimY, long dimZ)
{
    int       counter;
    long      k_m;
    long      k1;
    size_t    index;

    index = (size_t)(dimX * dimY * k) + (size_t)(j * dimX + i);
    
    float*    _values;
    _values = (float*) calloc(window_fulllength, sizeof(float));    
    
    counter = 0;
    for(k_m = -window_halflength_vertical; k_m <= window_halflength_vertical; k_m++)
    {
        k1 = k + k_m;
        if((k1 < 0) || (k1 >= dimZ))
            k1 = k-k_m;
        _values[counter] = input[(size_t)(dimX * dimY * k1) + (size_t)(j * dimX + i)];
        counter++;
    }
    quicksort_float(_values, 0, window_fulllength-1);
    output[index] = _values[midval_window_index];

    free (_values);
}


void
mean_stride3d(float* input, float* output,
                        long i, long j, long k,
                        long dimX, long dimY, long dimZ)
{
    /* a 3d mean to enusre a more stable gradient */
    long      i1;
    long      i2;
    long      j1;
    long      j2;
    long      k1;
    long      k2;    
    float     val1;
    float     val2;
    float     val3;
    float     val4;
    float     val5;
    float     val6;
    size_t    index;
  
    index = (size_t)(dimX * dimY * k) + (size_t)(j * dimX + i);

    i1 = i - 1;
    i2 = i + 1;
    j1 = j - 1;
    j2 = j + 1;
    k1 = k - 1;
    k2 = k + 1;

    if (i1 < 0)
        i1 = i2;
    if (i2 >= dimX)
        i2 = i1;
    if (j1 < 0)
        j1 = j2;
    if (j2 >= dimY)
        j2 = j1;  
    if (k1 < 0)
        k1 = k2;
    if (k2 >= dimZ)
        k2 = k1;

    val1 = input[(size_t)(dimX * dimY * k) + (size_t)(j * dimX + i1)];
    val2 = input[(size_t)(dimX * dimY * k) + (size_t)(j * dimX + i2)];
    val3 = input[(size_t)(dimX * dimY * k) + (size_t)(j1 * dimX + i)];
    val4 = input[(size_t)(dimX * dimY * k) + (size_t)(j2 * dimX + i)];
    val5 = input[(size_t)(dimX * dimY * k1) + (size_t)(j * dimX + i)];
    val6 = input[(size_t)(dimX * dimY * k2) + (size_t)(j * dimX + i)];
    
    output[index] = 0.1428f*(input[index] + val1 + val2 + val3 + val4 + val5 + val6);
}

void
remove_inconsistent_stripes(unsigned char* mask,
                            unsigned char* out, 
                            int stripe_length_min, 
                            int stripe_depth_min, 
                            float sensitivity,
                            long i,
                            long j,
                            long k,                            
                            long dimX, long dimY, long dimZ)
{
    int       counter_vert_voxels;    
    int       counter_depth_voxels;
    int       halfstripe_length = (int)stripe_length_min/2;
    int       halfstripe_depth = (int)stripe_depth_min/2;
    long      k_m;
    long      k1;
    long      y_m;
    long      y1;
    size_t    index;
    index = (size_t)(dimX * dimY * k) + (size_t)(j * dimX + i);

    int threshold_vertical = (int)((0.01f*sensitivity)*stripe_length_min);
    int threshold_depth = (int)((0.01f*sensitivity)*stripe_depth_min);

    counter_vert_voxels = 0;
    for(k_m = -halfstripe_length; k_m <= halfstripe_length; k_m++)
    {
        k1 = k + k_m;
         if((k1 < 0) || (k1 >= dimZ))
            k1 = k - k_m;
        if (mask[(size_t)(dimX * dimY * k1) + (size_t)(j * dimX + i)] == 1)
        {
            counter_vert_voxels++;
        }
    }
    
    /* Here we decide if to keep the currect voxel based on the number of vertical voxels bellow it */
    if (counter_vert_voxels > threshold_vertical)
    {
        /* 
        If the vertical non-zero values are consistent then the voxel could belong to a stripe.
        So we would like to check the depth consistency as well. Here we assume that the stripes 
        normally do not extend far in the depth dimension compared to the features that belong to a 
        sample. 
        */
       
       if (stripe_depth_min != 0)
       {
            counter_depth_voxels = 0;
            for(y_m = -halfstripe_depth; y_m <= halfstripe_depth; y_m++)
            {
                y1 = j + y_m;
                if((y1 < 0) || (y1 >= dimY))
                    y1 = j - y_m;
                if (mask[(size_t)(dimX * dimY * k) + (size_t)(y1 * dimX + i)] == 1)
                {
                    counter_depth_voxels++;
                }        
            }
            if (counter_depth_voxels < threshold_depth)
            {
            out[index] = 1;
            }
            else
            {
            out[index] = 0;
            }
        }
        else 
        {
            out[index] = 1;
        }
    }
    else
    {
        out[index] = 0;
    }
}                            

void
merge_stripes(unsigned char* mask,
              unsigned char* out, 
              int stripe_width_min, 
              long i,
              long j,
              long k,
              long dimX, long dimY, long dimZ)
{

    long        x_m;
    long        x1;
    long        x2;
    long        x2_m;
    size_t    index;
    index = (size_t)(dimX * dimY * k) + (size_t)(j * dimX + i);    

    if (mask[index] == 1)    
    {
        /* merging stripes in the horizontal direction */
        for(x_m=stripe_width_min; x_m>=0; x_m--) {
            x1 = i + x_m;
            if (x1 >= dimX)
                x1 = i - x_m;
            if (mask[(size_t)(dimX * dimY * k) + (size_t)(j * dimX + x1)] == 1)
            /*the other end of the mask has been found, merge all values inbetween */
            {
              for(x2 = 0; x2 <= x_m; x2++) 
              {
                x2_m = i + x2;
                out[(size_t)(dimX * dimY * k) + (size_t)(j * dimX + x2_m)] = 1;
              }
              break;
            }
        }            

    }
}





