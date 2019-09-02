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

#include "chan_vese_core.h"
#include "utils.h"

/*
 * Input Parameters (from Python):
 * Output:
 */

float chan_vese_main(float *Input, float *mask, float *Phi_x, float *Phi_xy, float *Phi_xx, float *Phi_y, float *Phi_yx, float *Phi_yy, float dt, float eta, float epsilon, float alpha_in, float alpha_out, int iterationsNumb, int dimX, int dimY, int dimZ)
{
    //int i,j;
    //float *Phi_x=NULL, *Phi_xx=NULL, *Phi_xy=NULL, *Phi_y=NULL, *Phi_yx=NULL, *Phi_yy=NULL;

    if (dimZ == 1)  {
    /*allocating space*/
    /*
    Phi_x = (float*) calloc (dimX*dimY, sizeof(float));
    Phi_xx = (float*) calloc (dimX*dimY, sizeof(float));
    Phi_xy = (float*) calloc (dimX*dimY, sizeof(float));

    Phi_y = (float*) calloc (dimX*dimY, sizeof(float));
    Phi_yx = (float*) calloc (dimX*dimY, sizeof(float));
    Phi_yy = (float*) calloc (dimX*dimY, sizeof(float));
    */

    /*Compute the  derivatives*/
    Gradient2D_central(mask, Phi_x, Phi_y, dimX, dimY);
    Gradient2D_central(Phi_x, Phi_xx, Phi_xy, dimX, dimY);
    Gradient2D_central(Phi_y, Phi_yx, Phi_yy, dimX, dimY);

    }
    /*
    free(Phi_x);
    free(Phi_xy);
    free(Phi_xx);
    free(Phi_y);
    free(Phi_yx);
    free(Phi_yy);
    */
    return 1;
}
/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
