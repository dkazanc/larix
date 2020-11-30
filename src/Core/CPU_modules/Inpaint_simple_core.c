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
 * 3. sigma - controlling parameter to start inpainting
 *
 * Output:
 * [1] Inpainted image/volume
 */

int Inpaint_simple_CPU_main(float *Input, unsigned char *Mask, float *Output, unsigned char *M_upd, int iterations, float sigma, int dimX, int dimY, int dimZ)
{
    long i, j, counter, countmask;

    /* copy into output */
    copyIm(Input, Output, (long)(dimX), (long)(dimY), (long)(dimZ));
    /* copying M to Mask_upd */
    copyIm_unchar(Mask, M_upd, dimX, dimY, dimZ);

    countmask = 0;
    for (i=0; i<dimY*dimX*dimZ; i++) if (M_upd[i] == 1) countmask++;

    if (countmask == 0) printf("%s \n", "Nothing to inpaint, zero mask!");
    else {
    for (i=0; i<iterations; i++) {
    if (dimZ == 1) linearcomb_inp_2D(Input, M_upd, Output, sigma, (long)(dimX), (long)(dimY)); /* running 2D inpainting */
    else linearcomb_inp_3D(Input, M_upd, Output, sigma, (long)(dimX), (long)(dimY), (long)(dimZ)); /* 3D version */
    counter = 0;
    for (j=0; j<dimY*dimX*dimZ; j++) if (M_upd[j] == 1) counter++;
    if (counter == 0) break;
     }
	 }
    return 0;
}
/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
/* inpainting using averaged interface values */
void linearcomb_inp_2D(float *Input, unsigned char *M_upd, float *Output, float sigma, long dimX, long dimY)
{
	long i, j, i_m, j_m, i1, j1, index, counter, W_halfsize;
	float sum_val;

 W_halfsize = 1;
#pragma omp parallel for shared(Input,M_upd,Output) private(index,i,j,i1,j1,i_m,j_m,counter,sum_val)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            index = j*dimX+i;
            if (M_upd[index] == 1) {
            counter = 0; sum_val = 0.0f;
            for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) {
                i1 = i+i_m;
                for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) {
                    j1 = j+j_m;
                    if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
                        if ((M_upd[j1*dimX + i1] == 0) && Output[j1*dimX + i1] > sigma) {
                        sum_val += Output[j1*dimX + i1];
                        counter++;
                       }}
                }}
                if (counter >= 3) {
                Output[index] = sum_val/counter;
                M_upd[index] = 0;
                }
            }
		}}
	return;
}

/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/
/* inpainting using averaged interface values */
void linearcomb_inp_3D(float *Input, unsigned char *M_upd, float *Output, float sigma, long dimX, long dimY, long dimZ)
{
	long i, j, k, i_m, j_m, k_m, i1, j1, k1, index, index2, counter, W_halfsize;
	float sum_val;

 W_halfsize = 1;
#pragma omp parallel for shared(Input,M_upd,Output) private(index, index2, i, j, k, i_m, j_m, k_m, i1, j1, k1,counter,sum_val)
for(k=0; k<dimZ; k++) {
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
            index = (dimX*dimY)*k + j*dimX+i;
            if (M_upd[index] == 1) {
            counter = 0; sum_val = 0.0f;
           for(k_m=-W_halfsize; k_m<=W_halfsize; k_m++) {
             k1 = k+k_m;
             for(i_m=-W_halfsize; i_m<=W_halfsize; i_m++) {
                i1 = i+i_m;
                for(j_m=-W_halfsize; j_m<=W_halfsize; j_m++) {
                    j1 = j+j_m;
                    index2 = (dimX*dimY)*k1 + j1*dimX+i1;
                    if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY)) && ((k1 >= 0) && (k1 < dimZ))) {
                        if ((M_upd[index2] == 0) && Output[index2] > sigma) {
                        sum_val += Output[index2];
                        counter++;
                       }}
                }}}
                if (counter >= 8) {
                Output[index] = sum_val/counter;
                M_upd[index] = 0;
                }
            }
		}}}
	return;
}
