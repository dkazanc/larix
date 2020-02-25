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

#include "autocropper_core.h"
#include "utils.h"

/* A data cropping algorithm where the object of interest lies within the FOV. 
 * The algorithm is developed mainly for cropping tomographic projection data
 *
 * Input Parameters (from Python):
 * Output: 
 */


float autocropper_main(float *Input, float *mask_box, float *crop_indeces, int margin_size, int statbox_size, int dimX, int dimY, int dimZ)
{
    //long i;    

    stat_collector2D(Input, mask_box, statbox_size, dimX, dimY);    
    return *mask_box; 
}

/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
float stat_collector2D(float *Input, float *mask_box, int statbox_size, int dimX, int dimY)
{
    /* the module places a box in the background region of the data in order to collect statistics */
    int statbox_size_vert, mid_vert_index_Y;
    long i, j;

    mid_vert_index_Y = (int)(0.5f*dimY);
    statbox_size_vert = (int)(2.5f*statbox_size);
    /* place the box in the vertical middle of the image */
    
    vertBox_index_up = mid_vert_index_Y-statbox_size_vert;
    vertBox_index_down = mid_vert_index_Y+statbox_size_vert;   

    for(j=0; j<dimY; j++) {
        if ((j >= vertBox_index_up) && (j <= vertBox_index_down)) {
        for(i=0; i<dimX; i++) {        
            if ((i >= 0) && (i <= statbox_size)) {
            mask_box[j*dimX+i] = 1.0;
            }
        }        
    }}
    return *mask_box;
}
