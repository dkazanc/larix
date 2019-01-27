#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"

/* C-OMP implementation of ...
 *
 *
 * http://www.lejordet.com/2009/04/simple-python-ctypes-example-in-windows/
 */

void flatsreg_main(unsigned short *flat, unsigned short *proj, int x, int y, int x1, int y1, int drift_window, int dimX, int dimY, float *error_vec) 
{
/*printf("Hello \n");*/
int i, j, i_m, j_m, i1, j1;
int size_p_x, size_p_y, i_new, j_new, i_flat, j_flat, drift_window_full;
float error_sum; 

drift_window_full = (int)(2*drift_window + 1);
//error_vec_size = pow((2*drift_window + 1),2);
size_p_x = (x1 - x);
size_p_y = (y1 - y);

//flat_patch_size = (size_p_x)*(size_p_y);
//flat_patch = (float*) calloc(flat_patch_size, sizeof(float));

#pragma omp parallel for shared(flat, proj, error_vec) private(i, j, i_m, j_m, error_sum, i1, j1, i_new, j_new, i_flat, j_flat)
 for(i_m=-drift_window; i_m<=drift_window; i_m++) {
     i = x + i_m;
      for(j_m=-drift_window; j_m<=drift_window; j_m++) {
      j = y + j_m;
      error_sum = 0.0f;
      for(i1=0; i1<size_p_x; i1++) {
          i_new = i + i1; /* moved patch on projection image */
          i_flat = x + i1; /*stationary patch of flats */
         for(j1=0; j1<size_p_y; j1++) {
            j_new = j + j1;
            j_flat = y + j1;
         error_sum += powf((float)(flat[j_flat*dimX + i_flat] - proj[j_new*dimX + i_new]),2);
         }}
      error_vec[(j_m+drift_window)*drift_window_full + (i_m + drift_window)] = sqrtf(error_sum/(size_p_x*size_p_y*drift_window_full*drift_window_full));
      }}
}


