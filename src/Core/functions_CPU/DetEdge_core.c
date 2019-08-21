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

#include "DetEdge_core.h"
#include "utils.h"

/*
 * Input Parameters (from Python):
 * Output:
 */

#ifndef max
    #define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
    #define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif


float Detect_edges_main(float *Input, unsigned char *output_mask, float *test_output, int LineSize, float threshold, int OrientNo, int dimX, int dimY, int dimZ)
{
    long i,j,index;
    int counterG, switcher, switchpad_crop, padDims, k;
    long DimTotal, dimX_pad, dimY_pad;
    float derrivativeX, derrivativeY, angle_degr, *Input_pad=NULL, *Rotate_pad=NULL, *Angles_Ar=NULL, *Output_mask_pad=NULL, df, ang_step;
    //unsigned char *Output_mask_pad=NULL;

    DimTotal = (long)(dimX*dimY*dimZ);
    Angles_Ar = (float*) calloc (OrientNo,sizeof(float));

    ang_step = (89.99f)/(float)(OrientNo);
    angle_degr = 0.0f;
    for (k=0; k < OrientNo; k++) {
        Angles_Ar[k] = angle_degr*(M_PI/180.0f);
        angle_degr += ang_step;
        //printf("[%f]\n", Angles_Ar[k]);
    }

    if (dimZ == 1) {

    if (dimX > dimY) {
      df = sqrtf(2.0f)*dimX;
      padDims = floor(df + 0.5f);
      padDims = (int)(padDims - dimX);
    }
    else {
      df = sqrtf(2.0f)*dimY;
      padDims = floor(df + 0.5f);
      padDims = (int)(padDims - dimY);
    }

    if (padDims%2 != 0) padDims = padDims + 1;
    dimX_pad = dimX + padDims;
    dimY_pad = dimY + padDims;

    /*allocating space*/
    Input_pad = (float*) calloc (dimX_pad*dimY_pad, sizeof(float));
    Rotate_pad = (float*) calloc (dimX_pad*dimY_pad*OrientNo, sizeof(float));
    //Output_mask_pad = (unsigned char*) calloc (dimX_pad*dimY_pad, sizeof(unsigned char));
    Output_mask_pad = (float*) calloc (dimX_pad*dimY_pad, sizeof(float));

    /* Padding input image with zeros to avoid loosing the data */
    switchpad_crop = 0; /*padding*/
    Pad_Crop(Input, Input_pad, dimX_pad, dimY_pad, padDims, switchpad_crop);

    #pragma omp parallel for shared (Input_pad, Rotate_pad) private(k)
    for (k=0; k < OrientNo; k++) {
      /* Rotate the padded image intro 3D array (N x M x Orientations)*/
      RotateImage(Input_pad, Rotate_pad, dimX_pad, dimY_pad, Angles_Ar[k], k);
    }

    //#pragma omp parallel for shared (Output_mask_pad, Rotate_pad) private(k)
    //for (k=0; k < OrientNo; k++) {
    k = 15;
    Proc_Rot_Array(Rotate_pad, Output_mask_pad, LineSize, dimX_pad, dimY_pad, OrientNo, threshold, k);
    //}

    switchpad_crop = 1; /*cropping*/
    Pad_Crop(Output_mask_pad, test_output, dimX_pad, dimY_pad, padDims, switchpad_crop);
    }
    free(Input_pad);
    free(Rotate_pad);
    free(Angles_Ar);
    free(Output_mask_pad);
    return 1;
}
/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
/* Function to check consistency of edges, the straight edges will be encouraged
We're looking for a smallest response in a choosen direction
*/
float Proc_Rot_Array(float *Rotate_pad, float *Output_mask_pad, int LineSize, int dimX, int dimY, int OrientNo, float threshold, int k)
{
  int i, j, l, l_i, l_j, w, l_i_w, l_j_w;
  float horiz_resp, vert_resp;
  for(j=0; j<dimY; j++) {
      for(i=0; i<dimX; i++) {
        /* moving horizontal and vertical lines over rotated image  */
        horiz_resp = 0.0f; vert_resp = 0.0f;
        for(l=-LineSize; l<LineSize; l++) {
          l_i = i + l;
          l_j = j + l;
            for(w=-LineSize; w<LineSize; w++) {
              l_i_w = i + w;
              /* do horizontal checking */
              if ((l_i >= 0) && (l_i < dimX) && (l_i_w >= 0) && (l_i_w < dimX)) {
              horiz_resp += fabs(Rotate_pad[dimX*dimY*k + j*dimX+l_i_w] - Rotate_pad[dimX*dimY*k + j*dimX+l_i]);
              }}
            for(w=-LineSize; w<LineSize; w++) {
                l_j_w = j + w;
              /* do vertical checking */
            if ((l_i >= 0) && (l_i < dimX) && (l_i_w >= 0) && (l_i_w < dimX)) {
              vert_resp += fabs(Rotate_pad[dimX*dimY*k + l_j_w*dimX+i] - Rotate_pad[dimX*dimY*k + l_j*dimX+i]);
            }}
          }
          if (vert_resp < horiz_resp) Output_mask_pad[j*dimX+i] = vert_resp;
          else Output_mask_pad[j*dimX+i] = horiz_resp;
      }}
}
