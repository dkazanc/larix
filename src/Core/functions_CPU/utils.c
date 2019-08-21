/*
Copyright 2019 Daniil Kazantsev & Diamond Light Source ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "utils.h"
#include <math.h>

/* Copy Image (float) */
float copyIm(float *A, float *U, long dimX, long dimY, long dimZ)
{
	long j;
#pragma omp parallel for shared(A, U) private(j)
	for (j = 0; j<dimX*dimY*dimZ; j++)  U[j] = A[j];
	return *U;
}

/* Copy Image */
unsigned char copyIm_unchar(unsigned char *A, unsigned char *U, int dimX, int dimY, int dimZ)
{
	int j;
#pragma omp parallel for shared(A, U) private(j)
	for (j = 0; j<dimX*dimY*dimZ; j++)  U[j] = A[j];
	return *U;
}

/*Roll image symmetrically from top to bottom*/
float copyIm_roll(float *A, float *U, int dimX, int dimY, int roll_value, int switcher)
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
    return *U;
}

/* image rotation using bilinear interpolation providing angle_rad */
float RotateImage(float *A, float *B, int dimX, int dimY, float angle_rad, int k)
{
	int i, j, i1, j1, l, m, i0, j0;
	float xx, yy, ll, mm, Ar1step_inv, Ar2step_inv, u, v, a, b, c, d, ct, st;
	float H_x, H_y, stepAR_X, stepAR_Y, Tomorange_Xmax, Tomorange_Xmin, Tomorange_Ymax, Tomorange_Ymin;
	Tomorange_Xmin = -1.0; Tomorange_Xmax = 1.0; Tomorange_Ymin = -1.0;  Tomorange_Ymax = 1.0;

	H_x = (float)(Tomorange_Xmax - Tomorange_Xmin)/(dimX-1);
	H_y = (float)(Tomorange_Ymax - Tomorange_Ymin)/(dimY-1);

	ct = cosf(angle_rad);
	st = sinf(angle_rad);

	Ar1step_inv = 1.0/H_x;
	Ar2step_inv = 1.0/H_y;

	for (l=0; l < dimX; l++) {
		stepAR_X = Tomorange_Xmin + l*H_x;
			for (m=0; m < dimY; m++) {
			stepAR_Y = Tomorange_Ymin + m*H_y;

			B[(dimX*dimY)*k + m*dimX+l] = 0.0;

			xx = stepAR_X*ct - stepAR_Y*st;
			yy = stepAR_X*st + stepAR_Y*ct;

			/*Bilinear 2Dim Interpolation */
			ll = (xx-Tomorange_Xmin)*Ar1step_inv;
			mm = (yy-Tomorange_Ymin)*Ar2step_inv;

			/*indexes calculation*/
			i0 = floor(ll);
			j0 = floor(mm);

			u = (float)(ll - i0);
			v = (float)(mm - j0);

			i=i0;
			j=j0;

			i1 = i+1;
			j1 = j+1;

			 /*Interpolation with border cases */
			a = 0.0; b = 0.0; c = 0.0; d = 0.0;
			if ((i > 0) && (i < dimX-1)) {
					if ((j > 0) && (j < dimY-1)) {
							a = A[j*dimX+i];
							b = A[j1*dimX+i];
							c = A[j*dimX+i1];
							d = A[j1*dimX+i1];
					}}
							B[(dimX*dimY)*k + m*dimX+l] = (1.0 - u)*(1.0 - v)*a + u*(1.0 - v)*b + (1.0 - u)*v*c + u*v*d;
			}
	}
	return *B;
}

/* pads (switchpad_crop = 0) or crops (switchpad_crop = 1) the image providing the padDims value */
float Pad_Crop(float *A, float *B, int dimX, int dimY, int padDims, int switchpad_crop) {

	int padDims2, i,j;

	padDims2 = (int)(0.5*padDims);
	for (j=0; j < dimY; j++) {
		for (i=0; i < dimX; i++) {
					if (((i >= padDims2) && (i < dimX-padDims2)) &&  ((j >= padDims2) && (j < dimY-padDims2)))  {
					if (switchpad_crop == 0) B[j*dimX+i] = A[(j-padDims2)*(dimX-padDims)+(i-padDims2)]; /*do padding*/
					else B[(j-padDims2)*(dimX-padDims)+(i-padDims2)] = A[j*dimX+i]; /* do cropping */
					}
		}}
	return *B;
}
