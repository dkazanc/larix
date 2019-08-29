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
    long i,j;
    int switchpad_crop, padDims, k;
    long dimX_pad, dimY_pad;
    float angle_degr, *Input_pad=NULL, *Rotate_pad=NULL, *Angles_Ar=NULL, *Output_mask_pad=NULL, df, ang_step;
    float *Output_minResid=NULL;
    //unsigned char *Output_mask_pad=NULL;

    //DimTotal = (long)(dimX*dimY*dimZ);
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
    Output_minResid = (float*) calloc (dimX_pad*dimY_pad*OrientNo*2, sizeof(float));
    Output_mask_pad = (float*) calloc (dimX_pad*dimY_pad, sizeof(float));

    /* Padding input image with zeros to avoid loosing the data */
    switchpad_crop = 0; /*padding*/
    Pad_Crop(Input, Input_pad, dimX_pad, dimY_pad, padDims, switchpad_crop);

    #pragma omp parallel for shared (Input_pad) private(k)
    for (k=0; k < OrientNo; k++) {
    /* Rotate the padded image intro 3D array (N x M x Orientations)*/
      RotateImage(Input_pad, Rotate_pad, dimX_pad, dimY_pad, Angles_Ar[k], k);
    }

    #pragma omp parallel for shared (Rotate_pad) private(k)
    for (k=0; k < OrientNo; k++) {
    Proc_Rot_Array(Rotate_pad, Output_minResid, LineSize, dimX_pad, dimY_pad, OrientNo, threshold, k);
    }

    float min_val, anglesel = 0.0f;
    for(j=0; j<dimY_pad; j++) {
        for(i=0; i<dimX_pad; i++) {
          min_val = Output_minResid[j*dimX_pad+i];
          for (k=0; k < 2*OrientNo; k++) {
            if (Output_minResid[dimY_pad*dimX_pad*k + j*dimX_pad+i] < min_val) min_val = Output_minResid[dimY_pad*dimX_pad*k + j*dimX_pad+i];
          }
          /*
          for (k=0; k < 2*OrientNo; k++) {
            if (Output_minResid[dimY_pad*dimX_pad*k + j*dimX_pad+i] == min_val) {
              if (k < OrientNo) {
                anglesel = min_val/Output_minResid[dimY_pad*dimX_pad*(k+OrientNo) + j*dimX_pad+i];
                //anglesel = Angles_Ar[k];
              }
              else {
                anglesel = min_val/Output_minResid[dimY_pad*dimX_pad*(k-OrientNo) + j*dimX_pad+i];
                //anglesel = Angles_Ar[k - OrientNo] + 0.5f*M_PI;
              }
            }
          }
          Output_mask_pad[j*dimX_pad+i] = anglesel;
          //Output_mask_pad[j*dimX_pad+i] = Output_minResid[dimY_pad*dimX_pad*4 + j*dimX_pad+i];
          */
          Output_mask_pad[j*dimX_pad+i] = min_val;
    }}

    switchpad_crop = 1; /*cropping*/
    Pad_Crop(Output_mask_pad, test_output, dimX_pad, dimY_pad, padDims, switchpad_crop);
  } // if (dimZ == 1)

    free(Input_pad);
    free(Rotate_pad);
    free(Angles_Ar);
    free(Output_mask_pad);
    free(Output_minResid);
    return 1;
}
/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
/*
Function to check consistency of edges, the straight edges will be encouraged
We're looking for a smallest response in a choosen direction
*/
float Proc_Rot_Array(float *Rotate_pad, float *Output_minResid, int LineSize, int dimX, int dimY, int OrientNo, float threshold, int k)
{
  /*
  int i, j, l, l_i, l_j, w, l_i_w, l_j_w;
  float horiz_resp, vert_resp;
  for(j=0; j<dimY; j++) {
      for(i=0; i<dimX; i++) {
        // moving horizontal and vertical lines over rotated image
        horiz_resp = 0.0f; vert_resp = 0.0f;
        for(l=-LineSize; l<LineSize; l++) {
          l_i = i + l;
          l_j = j + l;
            for(w=-LineSize; w<LineSize; w++) {
              l_i_w = i + w;
              // do horizontal checking
              if ((l_i >= 0) && (l_i < dimX) && (l_i_w >= 0) && (l_i_w < dimX)) {
              horiz_resp += powf((Rotate_pad[dimX*dimY*k + j*dimX+l_i_w] - Rotate_pad[dimX*dimY*k + j*dimX+l_i]),2);
              }}
            for(w=-LineSize; w<LineSize; w++) {
                l_j_w = j + w;
              // do vertical checking
            if ((l_i >= 0) && (l_i < dimX) && (l_i_w >= 0) && (l_i_w < dimX)) {
              vert_resp += powf((Rotate_pad[dimX*dimY*k + l_j_w*dimX+i] - Rotate_pad[dimX*dimY*k + l_j*dimX+i]),2);
            }}
          }
          if (vert_resp < horiz_resp) Output_minResid[dimX*dimY*k + j*dimX+i] = vert_resp;
          else Output_minResid[dimX*dimY*k + j*dimX+i] = horiz_resp;
      }}
      */
      int i, j, l, l_i, l_j;
      float horiz_resp, vert_resp, mean_resp_horiz, mean_resp_vert;
      for(j=0; j<dimY; j++) {
          for(i=0; i<dimX; i++) {
            // moving horizontal and vertical lines over rotated image
            mean_resp_horiz = 0.0f; mean_resp_vert  = 0.0f;
            for(l=-LineSize; l<LineSize; l++) {
              l_i = i + l;
              l_j = j + l;
                  // do horizontal checking
                  if ((l_i >= 0) && (l_i < dimX)) {
                  mean_resp_horiz += Rotate_pad[dimX*dimY*k + j*dimX+l_i]/(2.0*LineSize+1);
                  }
                  // do vertical checking
                if ((l_j >= 0) && (l_j < dimY)) {
                  mean_resp_vert += Rotate_pad[dimX*dimY*k + l_j*dimX+i]/(2.0*LineSize+1);
                }
              }
              horiz_resp = 0.0f; vert_resp = 0.0f;
              for(l=-LineSize; l<LineSize; l++) {
                l_i = i + l;
                l_j = j + l;
                    // do horizontal checking
                    if ((l_i >= 0) && (l_i < dimX)) {
                    horiz_resp += powf((Rotate_pad[dimX*dimY*k + j*dimX+l_i] - mean_resp_horiz), 2);
                    }
                    // do vertical checking
                  if ((l_j >= 0) && (l_j < dimY)) {
                    vert_resp += powf((Rotate_pad[dimX*dimY*k + l_j*dimX+i] - mean_resp_vert), 2);
                  }
                }

              Output_minResid[dimX*dimY*k + j*dimX+i] = horiz_resp/(2.0*LineSize+1);
              Output_minResid[dimX*dimY*(k+OrientNo) + j*dimX+i] = vert_resp/(2.0*LineSize+1);
          }}
      return *Output_minResid;
}
