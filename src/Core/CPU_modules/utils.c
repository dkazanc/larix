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

void sort_quick(float *x, int left_idx, int right_idx)
{
      float temp;
      int i = left_idx, j = right_idx;
      float pivot = x[(left_idx + right_idx) / 2];
      while (i <= j)
      {
            while (x[i] < pivot)
                  i++;
            while (x[j] > pivot)
                  j--;
            if (i <= j) {
                  temp = x[i];
                  x[i] = x[j];
                  x[j] = temp;
                  i++;
                  j--;
            }
      };
      if (left_idx < j)
            sort_quick(x, left_idx, j);
      if (i < right_idx)
            sort_quick(x, i, right_idx);
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
