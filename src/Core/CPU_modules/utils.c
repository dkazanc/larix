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

#include "utils.h"
#include <math.h>

void swap(float *xp, float *yp)
{
    float temp = *xp;
    *xp = *yp;
    *yp = temp;
}

int signum(int i) {
    return (i>0)?1:((i<0)?-1:0);
}


/* Copy Image (float) */
void copyIm(float *A, float *U, long dimX, long dimY, long dimZ)
{
	long j;
#pragma omp parallel for shared(A, U) private(j)
	for (j = 0; j<dimX*dimY*dimZ; j++)  U[j] = A[j];
	return;
}

/* Copy Image -unsigned char (8bit)*/
void copyIm_unchar(unsigned char *A, unsigned char *U, int dimX, int dimY, int dimZ)
{
	int j;
#pragma omp parallel for shared(A, U) private(j)
	for (j = 0; j<dimX*dimY*dimZ; j++)  U[j] = A[j];
	return;
}

/* Copy Image -unsigned char long long (8bit)*/
void copyIm_unchar_long(unsigned char *A, unsigned char *U, long long totalvoxels)
{
    size_t j;
#pragma omp parallel for shared(A, U) private(j)
	for (j = 0; j<totalvoxels; j++)  U[j] = A[j];
	return;
}

/* Copy Image - unsigned short (16bit)*/
void copyIm_unshort(unsigned short *A, unsigned short *U, int dimX, int dimY, int dimZ)
{
	int j;
#pragma omp parallel for shared(A, U) private(j)
	for (j = 0; j<dimX*dimY*dimZ; j++)  U[j] = A[j];
	return;
}


/*Roll image symmetrically from top to bottom*/
void copyIm_roll(float *A, float *U, int dimX, int dimY, int roll_value, int switcher)
{
    int i, j;
#pragma omp parallel for shared(U, A) private(i,j)
    for (i=0; i<dimX; i++) {
        for (j=0; j<dimY; j++) {
            if (switcher == 0) {
                if (j < (dimY - roll_value)) U[j*dimX + i] = A[(j+roll_value)*dimX + i];
                else U[j*dimX + i] = A[(j - (dimY - roll_value))*dimX + i];
            }
            else {
                if (j < roll_value) U[j*dimX + i] = A[(j+(dimY - roll_value))*dimX + i];
                else U[j*dimX + i] = A[(j - roll_value)*dimX + i];
            }
        }}
    return;
}

/* sorting using bubble method (float)*/
void sort_bubble_float(float *x, int n_size)
{
	int i,j;
	float temp;

	for (i = 0; i < n_size - 1; i++)
	{
		for(j = 0; j < n_size - i - 1; j++)
		{
			if (x[j] > x[j+1])
			{
				temp = x[j];
				x[j] = x[j+1];
				x[j+1] = temp;
			}
		}
	}
    return;
}

/* sorting using bubble method (uint16)*/
void sort_bubble_uint16(unsigned short *x, int n_size)
{
	int i,j;
	unsigned short temp;

	for (i = 0; i < n_size - 1; i++)
	{
		for(j = 0; j < n_size - i - 1; j++)
		{
			if (x[j] > x[j+1])
			{
				temp = x[j];
				x[j] = x[j+1];
				x[j+1] = temp;
			}
		}
	}
    return;
}

void quicksort_float(float *x, int first, int last)
{
   int i, j, pivot;
   float temp;

   if(first<last){
      pivot=first;
      i=first;
      j=last;

      while(i<j){
         while(x[i]<=x[pivot]&&i<last)
            i++;
         while(x[j]>x[pivot])
            j--;
         if(i<j){
            temp=x[i];
            x[i]=x[j];
            x[j]=temp;
         }
      }

      temp=x[pivot];
      x[pivot]=x[j];
      x[j]=temp;
      quicksort_float(x,first,j-1);
      quicksort_float(x,j+1,last);

   }
   return;
}

void quicksort_uint16(unsigned short *x, int first, int last)
{
   int i, j, pivot;
   unsigned short temp;

   if(first<last){
      pivot=first;
      i=first;
      j=last;

      while(i<j){
         while(x[i]<=x[pivot]&&i<last)
            i++;
         while(x[j]>x[pivot])
            j--;
         if(i<j){
            temp=x[i];
            x[i]=x[j];
            x[j]=temp;
         }
      }

      temp=x[pivot];
      x[pivot]=x[j];
      x[j]=temp;
      quicksort_uint16(x,first,j-1);
      quicksort_uint16(x,j+1,last);

   }
   return;
}


void max_val_mask(float *Input, unsigned char *Mask, float *minmax_array, long dimX, long dimY, long dimZ)
{
    /* ____ getting a maximum and minimum values of the input array defined by MASK____
    in order to use this function one needs to initialise 1x2 size minmax_array:
    e.g.:
    float *minmax_array;
    minmax_array = (float*) calloc (2,sizeof(float));
    */
    long i, j, k, counter, index;
    float min_val, max_val;

    /* collecting statistics*/
    counter = 0; min_val = 0.0f; max_val=0.0f;
    for(k=0; k<dimZ; k++) {
      for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
          index = (dimX*dimY)*k + j*dimX+i;
          if ((Mask[index] == 1) && (counter == 1)) {
          if (Input[index] < min_val) min_val = Input[index];
          if (Input[index] > max_val) max_val = Input[index];  }
          if ((Mask[index] == 1) && (counter == 0)) {
            min_val = Input[index];
            max_val = Input[index];
            counter = 1; }
        }}}
    minmax_array[0] = min_val;
    minmax_array[1] = max_val;
    return;
}


void gradient2D(float *Input, float *Output, long dimX, long dimY, int axis, int gradient_gap)
{  /*calculate the derrivative of the 2D input in the "axis" direction using the defined gradient_gap between neighbouring pixels  */
    long i, j, i1, j1, index;
    #pragma omp parallel for shared(Input, Output) private(i,j,i1,j1,index)

    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
            index = j*dimX+i;
            /* Forward differences */
            if (axis == 1) {
                j1 = j + gradient_gap; 
                if (j1 >= dimY) j1 = j - gradient_gap;
                if (j1 < 0) j1 = j + gradient_gap;
                Output[index] = Input[j1*dimX + i] - Input[index]; /* x+ */
            }
            else {
                i1 = i + gradient_gap; if (i1 >= dimX) i1 = i-gradient_gap;    
                Output[index] = Input[j*dimX + i1] - Input[index]; /* y+ */
            }
        }}
}


void gradient3D(float *Input, float *Output, long dimX, long dimY, long dimZ, int axis, int gradient_gap)
{  /*calculate the derrivative of the 3D input in the "axis" direction using the defined gradient_gap between neighbouring pixels  */
    long i, j, k, i1, j1, k1, index;
    #pragma omp parallel for shared(Input, Output) private(i,j,k,i1,j1,k1,index)

    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
            for(k=0; k<dimZ; k++) {
            index = (dimX*dimY)*k + j*dimX+i;
            /* Forward differences */
            if (axis == 0) {
                i1 = i + gradient_gap; if (i1 >= dimX) i1 = i-gradient_gap;    
                Output[index] = Input[(dimX*dimY)*k + j*dimX+i1] - Input[index]; /* y+ */
            }
            else if (axis == 1) {
                j1 = j + gradient_gap; if (j1 >= dimY) j1 = j-gradient_gap;                
                Output[index] = Input[(dimX*dimY)*k + j1*dimX+i] - Input[index]; /* x+ */
            }
            else {
                k1 = k + gradient_gap; if (k1 >= dimZ) k1 = k-gradient_gap;    
                Output[index] = Input[(dimX*dimY)*k1 + j*dimX+i] - Input[index]; /* z+ */
            }
        }}}
}

void fill_vector_with_neigbours1D(float *Input, float *_values, int W_halfsizeY, long dimX, long dimY, long i, long j, long index)
{  /*fill the given vector with the values in the vertical neighbourhood of the pixel i,j */
    long j_m, j1, counter_local, index2;

    counter_local = 0;
    for(j_m=-W_halfsizeY; j_m<=W_halfsizeY; j_m++) 
    {
        j1 = j+j_m;
        if ((j1 >= 0) && (j1 < dimY)) 
        {
             index2 = j1*dimX + i;
            _values[counter_local] = Input[index2];
        }
        else _values[counter_local] = Input[index];
        counter_local++; 
    }
}


void fill_vector_with_neigbours2D(float *Input, float *_values, int W_halfsizeY, int W_halfsizeX, long dimX, long dimY, long i, long j)
{  /*fill the given vector with the values in the neighbourhood of the pixel i,j */
    long i_m, j_m, i1, j1, counter_local, index, index2;
    index = j*dimX + i;

    counter_local = 0;
    for(i_m=-W_halfsizeX; i_m<=W_halfsizeX; i_m++) {
        i1 = i+i_m;
        for(j_m=-W_halfsizeY; j_m<=W_halfsizeY; j_m++) {
            j1 = j+j_m;
            if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
                 index2 = j1*dimX + i1;
                _values[counter_local] = Input[index2];                
            }
            else _values[counter_local] = Input[index];
            counter_local++; 
        }}
}

void fill_vector_with_neigbours3D(float *Input, float *_values,  int W_halfsizeZ, int W_halfsizeX, int W_halfsizeY, long dimX, long dimY, long dimZ, long i, long j, long k)
{  /*fill the given vector with the values in the neighbourhood of the voxel i,j,k */
    long i_m, j_m, i1, j1, counter_local, index, index2;
    index = (dimX*dimY)*k + j*dimX + i;

    /*2D window in a loop, 3D version bellow needs refining!*/
    counter_local = 0;
    for(i_m=-W_halfsizeX; i_m<=W_halfsizeX; i_m++) {
        i1 = i+i_m;
        for(j_m=-W_halfsizeY; j_m<=W_halfsizeY; j_m++) {
            j1 = j+j_m;
            if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
                 index2 = (dimX*dimY)*k + j1*dimX + i1;
                _values[counter_local] = Input[index2];                
            }
            else _values[counter_local] = Input[index];
            counter_local++; 
        }}

    /*
    counter_local = 0;
    for(i_m=-W_halfsizeX; i_m<=W_halfsizeX; i_m++) {
        i1 = i+i_m;
        for(j_m=-W_halfsizeY; j_m<=W_halfsizeY; j_m++) {
            j1 = j+j_m;
            for(k_m=-W_halfsizeZ; k_m<=W_halfsizeZ; k_m++) {
                k1 = k+k_m;            
            if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY)) && ((k1 >= 0) && (k1 < dimZ))) {
                 index2 = (dimX*dimY)*k1 + j1*dimX + i1;
                _values[counter_local] = Input[index2];                
            }
            else _values[counter_local] = Input[index];
            counter_local++; 
        }}}
        */
}


void mask_dilate2D(unsigned char *input, unsigned char *output, long dimX, long dimY)
{
   /* dilating mask */
    long index, index2, j, i, i1, j1, i_m, j_m;

#pragma omp parallel for shared (input, output) private(index, j, i, i1, j1, i_m, j_m)
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
        index = j*dimX+i;

        if (input[index] == 1) {
          for(j_m=-1; j_m<=1; j_m++) {
            for(i_m=-1; i_m<=1; i_m++) {
                i1 = i+i_m;
                j1 = j+j_m;
                if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
                index2 = j1*dimX+i1; 
                output[index2] = 1;
                }
        }}
        }
    }}
    return;
}

void stats_calc(float *Input, float *Output, int take_abs, long dimX, long dimY, long dimZ)
{   /* this function calculates statistics of the input image and place it in a vector Output*/

    long i, j, k, index, dimall, midval;
    float min_val, max_val, mean_val;
    float *temp_input;
    dimall = (long)dimX*dimY*dimZ;
    midval = (long)(0.5f*dimall) - 1;

    temp_input = calloc(dimall, sizeof(float));

    /* collecting statistics*/
    mean_val = 0.0f;
    min_val = Input[0];
    max_val = Input[0];
    for(k=0; k<dimZ; k++) {
      for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
          index = (dimX*dimY)*k + j*dimX+i;          
          if (Input[index] < min_val) min_val = Input[index];
          if (Input[index] >= max_val) max_val = Input[index];            
          if (take_abs == 1) {
          mean_val+=fabs(Input[index]);
          temp_input[index] = fabs(Input[index]);}
          else {
            mean_val+=Input[index];
            temp_input[index] = Input[index];
          }
        }}}
    
    quicksort_float(temp_input, 0, dimall-1); 

    Output[0] = min_val;
    Output[1] = max_val;
    Output[2] = mean_val/(dimall-1);
    Output[3] = temp_input[midval];

    free(temp_input);
    return;
}