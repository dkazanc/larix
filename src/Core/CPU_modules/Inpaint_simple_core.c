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

#include "Inpaint_simple_core.h"
#include "utils.h"

/* A morphological inpainting scheme which progresses from the outer edge of the mask inwards, 
 * therefore, acting like a diffusion-type process
 *
 * Input Parameters:
 * 1. Image/volume to inpaint
 * 2. Mask of the same size as (1) in 'unsigned char' format  (ones mark the region to inpaint, zeros belong to the data)
 * 3. Iterations number - this is an additional number of iterations to run after the region has been inpainted
 * 4. Half-window size of the searching window
 * 5. method type to select an inpainting value: 0 - mean, 1 - median, 2 - random
 * 6. ncores - number of CPU threads to use (if given), 0 is the default value - using all available cores
 *
 * Output:
 * 1. Inpainted image or volume
 */
int Inpaint_simple_CPU_main(float *Input, unsigned char *Mask, float *Output, int iterations, int W_halfsize, int method_type, int ncores, int dimX, int dimY, int dimZ)
{
    long            i;
    long            j;
    long            k;
    long            l;
    long            m;
    long            i_m;
    long            j_m;
    long            k_m;
    long            counter;
    float           *Gauss_weights;
    float           *Updated;
    int             window_fullength;
    unsigned char   *M_upd;
    int             W_fullsize;
    size_t          countmask;
    size_t          iterations_mask_complete;
    size_t          totalvoxels;        

    if (ncores > 0) 
    {
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(ncores); // Use a number of threads for all consecutive parallel regions 
    }    

    totalvoxels = (size_t)(dimX) * (size_t)(dimY) * (size_t)(dimZ);
    Updated = malloc(totalvoxels * sizeof(float));
    M_upd = malloc(totalvoxels * sizeof(unsigned char));

    /********************PREPARATIONS***************************/

    /* copy input into output */
    memcpy(Output, Input, totalvoxels * sizeof(float));
    memcpy(Updated, Input, totalvoxels * sizeof(float));
    /* copying Mask to Mask_upd */
    memcpy(M_upd, Mask, totalvoxels * sizeof(unsigned char));

    /*calculate all nonzero values in the given mask */
    countmask = 0;
    for (m=0; m<totalvoxels; m++) 
    {
      if (Mask[m] == 1) 
      {
        Output[m] = 0.0f; /* prepeare output by zeroing values inside the mask */
        Updated[m] = 0.0f; /* prepeare output by zeroing values inside the mask */
        countmask++;
      }
    }

    /*full 1D dim of the similarity (neighbour) window */
    W_fullsize = (int)(2*W_halfsize + 1); 
    if (dimZ == 1) 
    {
      window_fullength = (int) (powf(W_fullsize, 2));
    }
    else 
    {
      window_fullength = (int) (powf(W_fullsize, 3));
    }
    /* initialise vector of gaussian weights */
    Gauss_weights = (float*)calloc(window_fullength, sizeof(float));

    /*  pre-calculation of Gaussian distance weights  */
    if (dimZ == 1) 
    {      
      counter = 0;
      for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) 
      {
        for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) 
        {
            Gauss_weights[counter] = expf(-(powf((i_m), 2) + powf((j_m), 2))/(2*window_fullength));
            counter++;
        }
      }
    }
    else 
    {      
      counter = 0;
      for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) 
      {
        for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) 
        {
          for(k_m=-W_halfsize; k_m<=W_halfsize; k_m++) 
          {
            Gauss_weights[counter] = expf(-(powf((i_m), 2) + powf((j_m), 2) + powf((k_m), 2))/(2*window_fullength));
            counter++;
          }
        }
      }
    }


    /* exit if nothing to inpaint */
    if (countmask == 0) 
    {
      free(Updated);
      free(M_upd);
      free(Gauss_weights);
      return 0;
    }
    /* The maximum number of required iterations to do the completion of the whole inpainted region */
    iterations_mask_complete = countmask;

    /********************************************************************/
    /****************2D version of the algorithm*************************/
    /********************************************************************/
    if (dimZ == 1) 
    {
      /* 1. Start iterations to inpaint the masked region first */
      for (l=0; l<iterations_mask_complete; l++) 
      {
        #pragma omp parallel for shared(M_upd, Output, Updated, Gauss_weights) private(i,j)
        for(i=0; i<dimX; i++) 
        {
          for(j=0; j<dimY; j++) 
          {
            if ((method_type == 1) || (method_type == 2)) 
            {
              median_rand_inpainting_2D(M_upd, Output, Updated, W_halfsize, window_fullength, method_type, i, j, (long)(dimX), (long)(dimY));
            }
            else 
            {
              eucl_weighting_inpainting_2D(M_upd, Output, Updated, Gauss_weights, W_halfsize, i, j, (long)(dimX), (long)(dimY)); 
            }
          }
        }
        memcpy(Output, Updated, totalvoxels * sizeof(float));

        /* check here if the iterations to complete the masked region should be terminated */
        countmask = 0;
        for (m=0; m<totalvoxels; m++) 
        {
          if (M_upd[m] == 1) 
          {
            countmask++;
          }
        }
        if (countmask == 0)
        {
          break; /*exit iterations_mask_complete loop */
        }
      }

      /* if random pixel assignment is selected, perform one smoothing operation at the last iteration to remove outliers */
      if (method_type == 2) 
      {
        memcpy(M_upd, Mask, totalvoxels * sizeof(unsigned char));
        #pragma omp parallel for shared(M_upd, Output, Updated) private(i,j)
        for(i=0; i<dimX; i++) 
        {
          for(j=0; j<dimY; j++) 
          {
            mean_smoothing_2D(M_upd, Output, Updated, i, j, (long)(dimX), (long)(dimY));
          }
        }
      }
      memcpy(Output, Updated, totalvoxels * sizeof(float));

      /* 2. The Masked region has already should be inpainted by now, initiate user-defined iterations
         This might be useful if some additional smoothing of the inpainted area is required */    
      if (iterations > 0) 
      {
        /*we need to reset M_upd mask for every iteration!*/
        memcpy(M_upd, Mask, totalvoxels * sizeof(unsigned char));
        for (l=0; l<iterations; l++) 
        {
          #pragma omp parallel for shared(M_upd, Output, Updated, Gauss_weights) private(i,j)
          for(i=0; i<dimX; i++) 
          {
            for(j=0; j<dimY; j++) 
            {
              if ((method_type == 1) || (method_type == 2)) 
              {
                median_rand_inpainting_2D(M_upd, Output, Updated, W_halfsize, window_fullength, method_type, i, j, (long)(dimX), (long)(dimY));
              }
              else {
                eucl_weighting_inpainting_2D(M_upd, Output, Updated, Gauss_weights, W_halfsize, i, j, (long)(dimX), (long)(dimY)); 
              }
            }
          } 
          memcpy(Output, Updated, totalvoxels * sizeof(float));
        }

        /* again for random method we remove outliers with local mean filter */
         if (method_type == 2) 
         {
           memcpy(M_upd, Mask, totalvoxels * sizeof(unsigned char));
           #pragma omp parallel for shared(Mask, Output, Updated) private(i,j)
           for(i=0; i<dimX; i++) 
           {
             for(j=0; j<dimY; j++) 
             {                
                mean_smoothing_2D(M_upd, Output, Updated, i, j, (long)(dimX), (long)(dimY));
             }
           }
           memcpy(Output, Updated, totalvoxels * sizeof(float));
         }        
      }
    } /*end of 2D case*/
    /********************************************************************/
    /****************3D version of the algorithm*************************/
    /********************************************************************/
    else 
    {
      /* 1. Start iterations to inpaint the masked region first */
      for (l=0; l<iterations_mask_complete; l++) 
      {
        #pragma omp parallel for shared(M_upd,Gauss_weights) private(i,j,k)
        for(i=0; i<dimX; i++) 
        {
          for(j=0; j<dimY; j++) 
          {
            for(k=0; k<dimZ; k++) 
            {
              if ((method_type == 1) || (method_type == 2)) 
              {
                median_rand_inpainting_3D(M_upd, Output, Updated, W_halfsize, window_fullength, method_type, i, j, k, (long)(dimX), (long)(dimY), (long)(dimZ));
              }
              else 
              {
                eucl_weighting_inpainting_3D(M_upd, Output, Updated, Gauss_weights, W_halfsize, i, j, k, (long)(dimX), (long)(dimY), (long)(dimZ));
              }
            }
          }
        }
        memcpy(Output, Updated, totalvoxels * sizeof(float));

        /* check here if the iterations to complete the masked region should be terminated */
        countmask = 0;
        for (m=0; m<totalvoxels; m++) 
        {
          if (M_upd[m] == 1) 
          {
            countmask++;
          }
        }
        if (countmask == 0)
        {
          break; /*exit iterations_mask_complete loop */
        }
      }

        /* if random pixel assignment is selected, perform one smoothing operation at the last iteration to remove outliers */
        if (method_type == 2) 
        {
          memcpy(M_upd, Mask, totalvoxels * sizeof(unsigned char));
          #pragma omp parallel for shared(M_upd, Output, Updated) private(i,j,k)
          for(i=0; i<dimX; i++) 
          {
            for(j=0; j<dimY; j++) 
            {
              for(k=0; k<dimZ; k++) 
              {              
                mean_smoothing_3D(M_upd, Output, Updated, i, j, k, (long)(dimX), (long)(dimY), (long)(dimZ));
              }
            }
          }
        }
        memcpy(Output, Updated, totalvoxels * sizeof(float));


        /* 2. The Masked region has already should be inpainted by now, initiate user-defined iterations
          This might be useful if some additional smoothing of the inpainted area is required */    
        if (iterations > 0) 
        {
          /*we need to reset M_upd mask for every iteration!*/
          memcpy(M_upd, Mask, totalvoxels * sizeof(unsigned char));
          for (l=0; l<iterations; l++) 
          {
          #pragma omp parallel for shared(M_upd, Output, Updated, Gauss_weights) private(i, j, k)
            for(i=0; i<dimX; i++) 
            {
              for(j=0; j<dimY; j++) 
              {
                for(k=0; k<dimZ; k++) 
                {
                    if ((method_type == 1) || (method_type == 2)) 
                    {
                      median_rand_inpainting_3D(M_upd, Output, Updated, W_halfsize, window_fullength, method_type, i, j, k, (long)(dimX), (long)(dimY), (long)(dimZ));
                    }
                    else 
                    {
                      eucl_weighting_inpainting_3D(M_upd, Output, Updated, Gauss_weights, W_halfsize, i, j, k, (long)(dimX), (long)(dimY), (long)(dimZ));
                    }
                }
              }
            }
          memcpy(Output, Updated, totalvoxels * sizeof(float));
          }
          
          /* again for random method we remove outliers with local mean filter */
          if (method_type == 2) 
          {
            memcpy(M_upd, Mask, totalvoxels * sizeof(unsigned char));
            #pragma omp parallel for shared(M_upd, Output, Updated) private(i,j,k)
            for(i=0; i<dimX; i++) 
            {
              for(j=0; j<dimY; j++) 
              {
                for(k=0; k<dimZ; k++) 
                {
                  mean_smoothing_3D(M_upd, Output, Updated, i, j, k, (long)(dimX), (long)(dimY), (long)(dimZ));
                }
              }
            }
          }
          memcpy(Output, Updated, totalvoxels * sizeof(float));
        }
    }

    free(Gauss_weights);
    free(Updated);
    free(M_upd);
    return 0;
}


int
floatcomp(const void* elem1, const void* elem2)
{
    if(*(const float*) elem1 < *(const float*) elem2)
        return -1;
    return *(const float*) elem1 > *(const float*) elem2;
}

/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
void mean_smoothing_2D(unsigned char *Mask, float *Output, float *Updated, long i, long j, long dimX, long dimY)
{
  long            i_m;
  long            j_m;
  long            i1;
  long            j1;
  float           sum_val;
  int             counter_local;
  size_t          index;
  size_t          index1;
  index = (size_t)j * dimX + (size_t)i;

  if (Mask[index] == 1) 
  {  
    counter_local = 0; sum_val = 0.0f;
    for(i_m=-1; i_m<=1; i_m++) 
    {
      i1 = i+i_m;
      for(j_m=-1; j_m<=1; j_m++) 
      {
        j1 = j+j_m;
        if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) 
        {
          index1 = (size_t)j1 * dimX + (size_t)i1;
          if (Output[index1] != 0.0) 
          {
          sum_val += Output[index1];
          counter_local++;
          }
        }
      }
    }            
    if (counter_local > 0) 
    {
    Updated[index] = sum_val/counter_local;
    }
  }
return;
}
void eucl_weighting_inpainting_2D(unsigned char *M_upd, float *Output, float *Updated, float *Gauss_weights, int W_halfsize, long i, long j, long dimX, long dimY)
{
  long            i_m;
  long            j_m;
  long            i1;
  long            j1;
  float           sum_val;
  float           sumweights;
  int             counter_local;
  int             counterglob;
  int             counter_vicinity;
  size_t          index;
  size_t          index1;
  index = (size_t)j * dimX + (size_t)i;

  /* check that you're on the region defined by the updated mask */
  if (M_upd[index] == 1) 
  {
      /* first check if there is usable information in the close vicinity of the mask's edge */
      counter_vicinity = 0;
      for(i_m=-1; i_m<=1; i_m++) 
      {
          i1 = i+i_m;
          for(j_m=-1; j_m<=1; j_m++) 
          {
              j1 = j+j_m;
              if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) 
              {
                if (Output[(size_t)j1 * dimX + (size_t)i1] != 0.0)
                {
                counter_vicinity++;
                break;
                }           
              }
          }
      }

      if (counter_vicinity > 0) 
      {
          counter_local = 0;
          sum_val = 0.0f;
          sumweights = 0.0f;
          counterglob = 0;
          for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) 
          {
            i1 = i+i_m;
              for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) 
              {
                j1 = j+j_m;
                  if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) 
                  {
                      index1 = (size_t)j1 * dimX + (size_t)i1;
                      if (Output[index1] != 0.0) 
                      {
                      sum_val += Output[index1]*Gauss_weights[counterglob];
                      sumweights += Gauss_weights[counterglob];
                      counter_local++;
                      }
                  }
                counterglob++;
              }
          }
          /* if there were non zero mask values */
          if (counter_local > 0) 
          {
          Updated[index] = sum_val/sumweights;
          M_upd[index] = 0;
          }
      }
    }
	return;
}

void median_rand_inpainting_2D(unsigned char *M_upd, float *Output, float *Updated, int W_halfsize, int window_fullength, int method_type, long i, long j, long dimX, long dimY)
{
  float           *_values;
  long            i_m;
  long            j_m;
  long            i1;
  long            j1;
  long            median_val;
  int             r0;
  int             r1;
  int             counter_local;
  float           vicinity_mean;
  size_t          index;
  size_t          index1;
  index = (size_t)j * dimX + (size_t)i;

  _values = (float*) calloc(window_fullength, sizeof(float));

  /* check that you're on the region defined by the updated mask */
  if (M_upd[index] == 1) 
  {
    /* a quick check if there is a usable information in the close vicinity of the mask's edge */
    counter_local = 0;
    vicinity_mean = 0.0f;
    for(i_m=-1; i_m<=1; i_m++) 
    {
        i1 = i+i_m;
        for(j_m=-1; j_m<=1; j_m++) 
        {
            j1 = j+j_m;
            if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) 
            {
              index1 = (size_t)j1 * dimX + (size_t)i1;
              if (Output[index1] != 0.0)
              {
              vicinity_mean += Output[index1];
              counter_local++;
              }
            }
        }
    }
    /*If we've got usable data in the vicinity then proceed with inpainting */
    if (vicinity_mean != 0.0f) 
    {
          vicinity_mean = vicinity_mean/counter_local; /* get the mean of values in the vicinity */
          /* fill the vectors */
          counter_local = 0;
          for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) 
          {
              i1 = i+i_m;
              for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) 
              {
                  j1 = j+j_m;
                  if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) 
                  {
                      index1 = (size_t)j1 * dimX + (size_t)i1;
                      if (Output[index1] != 0.0) 
                      {
                      _values[counter_local] = Output[index1];
                      counter_local++;
                      }
                  }
              }
          }              
      if (method_type == 1) 
      {
        /* inpainting based on the median neighbour (!can create sharp features!) */
        qsort(_values, counter_local-1, sizeof(float), floatcomp);
        median_val = (int)(counter_local/2);
        Updated[index] = _values[median_val];
      } 
      else 
      {
        /* inpainting based on a random neighbour (mean of two random values)*/
        r0 = rand() % counter_local;
        r1 = rand() % counter_local;
        Updated[index] = 0.5f * (_values[r0] + _values[r1]);
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
void mean_smoothing_3D(unsigned char *Mask, float *Output, float *Updated, long i, long j, long k, long dimX, long dimY, long dimZ)
{
    long            i_m;
    long            j_m;
    long            k_m;
    long            i1;
    long            j1;
    long            k1;
    float           sum_val;
    int             counter_local;
    size_t          index;
    size_t          index1;
    index = dimX * dimY * (size_t)k + (size_t)j * dimX + (size_t)i;

    if (Mask[index] == 1) 
    {  
      counter_local = 0;
      sum_val = 0.0f;
      for(i_m=-1; i_m<=1; i_m++) 
      {
        i1 = i+i_m;
        for(j_m=-1; j_m<=1; j_m++) 
        {
          j1 = j+j_m;
          for(k_m=-1; k_m<=1; k_m++) 
          {
            k1 = k+k_m;
            if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY)) && ((k1 >= 0) && (k1 < dimZ)))
            {
              index1 = dimX * dimY * (size_t)k1 + (size_t)j1 * dimX + (size_t)i1;
              if (Output[index1] != 0.0)
              {
              sum_val += Output[index1];
              counter_local++;
              }
            }
          }
        }
      }            
      if (counter_local > 0) 
      {
      Updated[index] = sum_val/counter_local;
      }
    }
  return;
}

void eucl_weighting_inpainting_3D(unsigned char *M_upd, float *Output,
                                  float *Updated, float *Gauss_weights,
                                  int W_halfsize, long i, long j, long k, 
                                  long dimX, long dimY, long dimZ)
{
    long            i_m;
    long            j_m;
    long            k_m;
    long            i1;
    long            j1;
    long            k1;
    float           sum_val;
    float           sumweights;
    long            counter_local;
    long            counterglob;
    long            counter_vicinity;
    size_t          index;
    size_t          index1;
    index = dimX * dimY * (size_t)k + (size_t)j * dimX + (size_t)i;

    /* check that you're on the region defined by the updated mask */
    if (M_upd[index] == 1) 
    {
      /* first check if there is usable information in the close vicinity of the mask's edge */
      counter_vicinity = 0;
      for(i_m=-1; i_m<=1; i_m++) 
      {
        i1 = i+i_m;
        for(j_m=-1; j_m<=1; j_m++) 
        {
          j1 = j+j_m;
          for(k_m=-1; k_m<=1; k_m++) 
          {
            k1 = k+k_m;
            if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY)) && ((k1 >= 0) && (k1 < dimZ))) 
            {
              if (Output[dimX * dimY * (size_t)k1 + (size_t)j1 * dimX + (size_t)i1] != 0.0)
              {
              counter_vicinity++;
              break;
              }
            }
          }
        }
      }

      if (counter_vicinity > 0)
      {
        /* there is data for inpainting -> proceed */
          counter_local = 0;
          sum_val = 0.0f;
          sumweights = 0.0f;
          counterglob = 0;
          for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) 
          {
            i1 = i+i_m;
              for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) 
              {
                j1 = j+j_m;
                  for(k_m=-W_halfsize; k_m<=W_halfsize; k_m++) 
                  {
                    k1 = k+k_m;              
                      if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY)) && ((k1 >= 0) && (k1 < dimZ))) 
                      {
                        index1 = dimX * dimY * (size_t)k1 + (size_t)j1 * dimX + (size_t)i1;
                        if (Output[index1] != 0.0)
                        {
                          sum_val += Output[index1]*Gauss_weights[counterglob];
                          sumweights += Gauss_weights[counterglob];
                          counter_local++;
                        }
                      }
                    counterglob++;
                  }
              }
          }
          /* if there were non zero mask values */
          if (counter_local > 0) 
          {
          Updated[index] = sum_val/sumweights;
          M_upd[index] = 0;
          }
      }      
    }
	return;
}

void median_rand_inpainting_3D(unsigned char *M_upd, float *Output,
                               float *Updated, int W_halfsize, int window_fullength,
                               int method_type,
                               long i, long j, long k, long dimX, long dimY, long dimZ)
{
  float           *_values;
  long            i_m;
  long            j_m;
  long            k_m;
  long            i1;
  long            j1;
  long            k1;    
  float           vicinity_mean;
  int             counter_local;
  int             r0;
  int             r1;
  int             median_val;
  size_t          index;
  size_t          index1;

  index = dimX * dimY * (size_t)k + (size_t)j * dimX + (size_t)i;

  _values = (float*) calloc(window_fullength, sizeof(float));

  /* check that you're on the region defined by the mask */
  if (M_upd[index] == 1) 
  {
    /* check if have a usable information in the vicinity of the mask's edge*/
    counter_local = 0; 
    vicinity_mean = 0.0f;
    for(i_m=-1; i_m<=1; i_m++) 
    {
      i1 = i+i_m;
      for(j_m=-1; j_m<=1; j_m++) 
      {
        j1 = j+j_m;
        for(k_m=-1; k_m<=1; k_m++) 
        {
          k1 = k+k_m;
          if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY)) && ((k1 >= 0) && (k1 < dimZ))) 
          {
            index1 = dimX * dimY * (size_t)k1 + (size_t)j1 * dimX + (size_t)i1;
            if (Output[index1] != 0.0) 
            {
            vicinity_mean += Output[index1];
            counter_local++;
            }
          }
        }
      }
    }            
          
    /* We've got usable data in the vicinity os the neighbourhood then proceed with inpainting */
    if (vicinity_mean != 0.0f) 
    {
      vicinity_mean = vicinity_mean/counter_local; /* get the mean of values in the vicinity */

      /* fill the vector */
      counter_local = 0;
        for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) 
        {
          i1 = i+i_m;
            for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) 
            {
              j1 = j+j_m;
                for(k_m=-W_halfsize; k_m<=W_halfsize; k_m++) 
                {
                  k1 = k+k_m;
                  if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY)) && ((k1 >= 0) && (k1 < dimZ))) 
                  {
                      index1 = dimX * dimY * (size_t)k1 + (size_t)j1 * dimX + (size_t)i1;
                      if (Output[index1] != 0.0) 
                      {
                        _values[counter_local] = Output[index1];
                        counter_local++;
                      }
                  }
                }
            }
        }
        if (method_type == 1) 
        {
          /* inpainting based on the median neighbour */
          qsort(_values, window_fullength, sizeof(float), floatcomp);
          median_val = (int)(counter_local/2);
          Updated[index] = _values[median_val];
        }
        else 
        {
          /* inpainting based on two random neighbours */
          r0 = rand() % counter_local;
          r1 = rand() % counter_local;
          Updated[index] = 0.5f *(_values[r0] + _values[r1]);
        }
        M_upd[index] = 0;
    }
  }
  free(_values);
	return;
}