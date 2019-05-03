/*
 * This work is part of the Core Imaging Library developed by
 * Visual Analytics and Imaging System Group of the Science Technology
 * Facilities Council, STFC
 *
 * Copyright 2019 Daniil Kazantsev
 * Copyright 2019 Srikanth Nagella, Edoardo Pasca
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

#include "MASK_merge_core.h"
#include "utils.h"

/* A method to ensure connectivity within regions of the segmented image/volume. Here we assume
 * that the MASK has been obtained using some classification/segmentation method such as k-means or gaussian
 * mixture. Some pixels/voxels have been misclassified and we check the spatial dependences
 * and correct the mask. We check the connectivity using the bresenham line algorithm within the non-local window
 * surrounding the pixel of interest.
 *
 * Input Parameters:
 * 1. MASK [0:255], the result of some classification algorithm (information-based preferably)
 * 2. The list of classes (e.g. [3,4]) to apply the method. The given order matters.
 * 3. The total number of classes in the MASK.
 * 4. The size of the Correction Window inside which the method works.

 * Output:
 * 1. MASK_upd - the UPDATED MASK where some regions have been corrected (merged) or removed
 * 2. CORRECTEDRegions - The array of the same size as MASK where all regions which were
 * changed are highlighted and the changes have been counted
 */

float Mask_merge_main(unsigned char *MASK, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *SelClassesList, unsigned char *ComboClasses, int tot_combinations, int SelClassesList_length, int classesNumb, int CorrectionWindow, int iterationsNumb, int dimX, int dimY, int dimZ)
{
    long i,j,k,l;
    int counterG, switcher;
    long DimTotal;
    unsigned char *MASK_temp, *ClassesList, CurrClass, temp, class1, class2, class3;
    DimTotal = (long)(dimX*dimY*dimZ);

    /* defines the list for all classes in the mask */
    ClassesList = (unsigned char*) calloc (classesNumb,sizeof(unsigned char));

     /* find which classes (values) are present in the segmented data */
     CurrClass =  MASK[0]; ClassesList[0]= MASK[0]; counterG = 1;
     for(i=0; i<DimTotal; i++) {
       if (MASK[i] != CurrClass) {
          switcher = 1;
          for(j=0; j<counterG; j++) {
            if (ClassesList[j] == MASK[i]) {
              switcher = 0;
              break;
            }}
            if (switcher == 1) {
                CurrClass = MASK[i];
                ClassesList[counterG] = MASK[i];
                /*printf("[%u]\n", ClassesList[counterG]);*/
                counterG++;
              }
        }
        if (counterG == classesNumb) break;
      }
      /* sort from LOW->HIGH the obtained values (classes) */
      for(i=0; i<classesNumb; i++)	{
                  for(j=0; j<classesNumb-1; j++) {
                      if(ClassesList[j] > ClassesList[j+1]) {
                          temp = ClassesList[j+1];
                          ClassesList[j+1] = ClassesList[j];
                          ClassesList[j] = temp;
                      }}}

    MASK_temp = (unsigned char*) calloc (DimTotal,sizeof(unsigned char));

    /* copy given MASK to MASK_upd*/
    copyIm_unchar(MASK, MASK_upd, (long)(dimX), (long)(dimY), (long)(dimZ));

    if (dimZ == 1) {
    /* start iterations */
    for(k=0; k<iterationsNumb; k++) {

    /********************** PERFORM 2D MASK PROCESSING ************************/
    #pragma omp parallel for shared(MASK,MASK_upd) private(i,j)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
    /* STEP1: in a smaller neighbourhood check that the current pixel is NOT an outlier */
    OutiersRemoval2D(MASK, MASK_upd, i, j, (long)(dimX), (long)(dimY));
    }}
    /* copy the updated MASK (clean of outliers) */
    copyIm_unchar(MASK_upd, MASK_temp, (long)(dimX), (long)(dimY), (long)(dimZ));

    // printf("[%u][%u][%u]\n", ClassesList[0], ClassesList[1], ClassesList[2]);

    for(l=0; l<SelClassesList_length; l++) {
    /*printf("[%u]\n", ClassesList[SelClassesList[l]]);*/
    #pragma omp parallel for shared(MASK_temp,MASK_upd,l) private(i,j)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
      /* The class of the central pixel has not changed, i.e. the central pixel is not an outlier -> continue */
      if (MASK_temp[j*dimX+i] == MASK[j*dimX+i]) {
	/* !One needs to work with a specific class to avoid overlaps! It is
	     crucial to establish relevant classes first (given as an input in SelClassesList) */
       if (MASK_temp[j*dimX+i] == ClassesList[SelClassesList[l]]) {
        Mask_update_main2D(MASK_temp, MASK_upd, CORRECTEDRegions, ClassesList, i, j, CorrectionWindow, (long)(dimX), (long)(dimY));
       	  }}
      }}
      /* copy the updated mask */
      copyIm_unchar(MASK_upd, MASK_temp, (long)(dimX), (long)(dimY), (long)(dimZ));     
      } /*SelClassesList_length*/
       /* Main classes have been processed. Working with implausable combinations */
       /* loop over the combinations of 3 */
      
       for(l=0; l<tot_combinations; l++) {
	 class1 = ComboClasses[l*3];
	 class2 = ComboClasses[l*3+1];
	 class3 = ComboClasses[l*3+2];
	 printf("[%u][%u][%u]\n", class1, class2, class3);
	 /*
	 #pragma omp parallel for shared(MASK_temp,MASK_upd, l, class1, class2, class3) private(i,j)
	     for(i=0; i<dimX; i++) {
	             for(j=0; j<dimY; j++) { 
		        Mask_update_combo2D(MASK_temp, MASK_upd, ClassesList, class1, class2, class3, i, j, CorrectionWindow, (long)(dimX), (long)(dimY));        
		}}		        
         copyIm_unchar(MASK_upd, MASK_temp, (long)(dimX), (long)(dimY), (long)(dimZ));            
         */
       }             
      }
    }
    else {
    /********************** PERFORM 3D MASK PROCESSING ************************/

    }
    free(MASK_temp);
    free(ClassesList);
    return *MASK_upd;
}
/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
float OutiersRemoval2D(unsigned char *MASK, unsigned char *MASK_upd, long i, long j, long dimX, long dimY)
{
  /*if the ROI pixel does not belong to the surrondings, turn it into the surronding*/
  long i_m, j_m, i1, j1, counter;
    counter = 0;
    for(i_m=-1; i_m<=1; i_m++) {
      for(j_m=-1; j_m<=1; j_m++) {
        i1 = i+i_m;
        j1 = j+j_m;
        if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
          if (MASK[j*dimX+i] != MASK[j1*dimX+i1]) counter++;
        }
      }}
      if (counter >= 8) MASK_upd[j*dimX+i] = MASK[j1*dimX+i1];
      return *MASK_upd;
}

float Mask_update_main2D(unsigned char *MASK_temp, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *ClassesList, long i, long j, int CorrectionWindow, long dimX, long dimY)
{
  long i_m, j_m, i1, j1, CounterOtherClass;

  /* STEP2: in a larger neighbourhood check that the other class is present  */
  CounterOtherClass = 0;
  for(i_m=-CorrectionWindow; i_m<=CorrectionWindow; i_m++) {
      for(j_m=-CorrectionWindow; j_m<=CorrectionWindow; j_m++) {
        i1 = i+i_m;
        j1 = j+j_m;
        if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
          if (MASK_temp[j*dimX+i] != MASK_temp[j1*dimX+i1]) CounterOtherClass++;
        }
      }}
      if (CounterOtherClass > 0) {
      /* the other class is present in the vicinity of CorrectionWindow, continue to STEP 3 */
      /*
      STEP 3: Loop through all neighbours in CorrectionWindow and check the spatial connection.
      Meaning that we're instrested if there are any classes between points A and B that
      does not belong to A and B (A,B \in C)
      */
      for(i_m=-CorrectionWindow; i_m<=CorrectionWindow; i_m++) {
          for(j_m=-CorrectionWindow; j_m<=CorrectionWindow; j_m++) {
            i1 = i+i_m;
            j1 = j+j_m;
            if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
              if (MASK_temp[j*dimX+i] == MASK_temp[j1*dimX+i1]) {
               /* A and B points belong to the same class, do STEP 4*/
               /* STEP 4: Run the Bresenham line algorithm between A and B points
               and convert all points on the way to the class of A. */
              bresenham2D(i, j, i1, j1, MASK_temp, MASK_upd, CORRECTEDRegions, ClassesList, 0, (long)(dimX), (long)(dimY));
             }
            }
          }}
      }
  return *MASK_upd;
}

float Mask_update_combo2D(unsigned char *MASK_temp, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *ClassesList, unsigned char class1, unsigned char class2, unsigned char class3, long i, long j, int CorrectionWindow, long dimX, long dimY)
{
  long i_m, j_m, i1, j1, CounterOtherClass;

  /* STEP2: in a larger neighbourhood check that the other class is present  */
  CounterOtherClass = 0;
  for(i_m=-CorrectionWindow; i_m<=CorrectionWindow; i_m++) {
      for(j_m=-CorrectionWindow; j_m<=CorrectionWindow; j_m++) {
        i1 = i+i_m;
        j1 = j+j_m;
        if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
          if (MASK_temp[j*dimX+i] != MASK_temp[j1*dimX+i1]) CounterOtherClass++;
        }
      }}
      if (CounterOtherClass > 0) {
      /* the other class is present in the vicinity of CorrectionWindow, continue to STEP 3 */
      /*
      STEP 3: Loop through all neighbours in CorrectionWindow and check the spatial connection.
      Meaning that we're instrested if there are any classes between points A and B that
      does not belong to A and B (A,B \in C)
      */
      for(i_m=-CorrectionWindow; i_m<=CorrectionWindow; i_m++) {
          for(j_m=-CorrectionWindow; j_m<=CorrectionWindow; j_m++) {
            i1 = i+i_m;
            j1 = j+j_m;
            if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
              if ((MASK_temp[j*dimX+i] == ClassesList[1]) && (MASK_temp[j1*dimX+i1] == ClassesList[3])) {
              /* points A and B belong to different classes (specifically 1 (loop) and 3 (liquor))! We consider
              the combination 1 -> 2 -> 3 (loop->crystal->liquor) is not plausable.  */
              bresenham2D(i, j, i1, j1, MASK_temp, MASK_upd, CORRECTEDRegions, ClassesList, 1, (long)(dimX), (long)(dimY));
              }
              if ((MASK_temp[j*dimX+i] == ClassesList[0]) && (MASK_temp[j1*dimX+i1] == ClassesList[3])) {
              /* improbabale combination 0 -> 4 -> 3 (air->artifacts->liquor): 4 -> 3  or
                 improbabale (!) combination 0 -> 1 -> 3 (air->loop->liquor): 1 -> 3  */
              bresenham2D(i, j, i1, j1, MASK_temp, MASK_upd, CORRECTEDRegions, ClassesList, 2, (long)(dimX), (long)(dimY));
              }
              if ((MASK_temp[j*dimX+i] == ClassesList[0]) && (MASK_temp[j1*dimX+i1] == ClassesList[1])) {
              /* improbabale combination 0 -> 4 -> 1 (air->artifacts->loop): 4 -> 1 or
                 improbabale combination 0 -> 2 -> 1 (air->crystal->loop): 2 -> 1 or  */
              bresenham2D(i, j, i1, j1, MASK_temp, MASK_upd, CORRECTEDRegions, ClassesList, 3, (long)(dimX), (long)(dimY));
              }
              if ((MASK_temp[j*dimX+i] == ClassesList[0]) && (MASK_temp[j1*dimX+i1] == ClassesList[2])) {
              /* improbabale combination 0 -> 1 -> 2 (air->loop->crystal): 1 -> 2 */
              bresenham2D(i, j, i1, j1, MASK_temp, MASK_upd, CORRECTEDRegions, ClassesList, 4, (long)(dimX), (long)(dimY));
              }
            }
          }}
      }
  return *MASK_upd;
}
int bresenham2D(int i, int j, int i1, int j1, unsigned char *MASK, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *ClassesList, int class_switcher, long dimX, long dimY)
{
                   int n;
                   int x[] = {i, i1};
                   int y[] = {j, j1};
                   int steep = (fabs(y[1]-y[0]) > fabs(x[1]-x[0]));
                   int ystep = 0;

                   //printf("[%i][%i][%i][%i]\n", x[1], y[1], steep, kk) ;
                   //if (steep == 1) {swap(x[0],y[0]); swap(x[1],y[1]);}

                   if (steep == 1) {
                   // swaping
                   int a, b;

                   a = x[0];
                   b = y[0];
                   x[0] = b;
                   y[0] = a;

                   a = x[1];
                   b = y[1];
                   x[1] = b;
                   y[1] = a;
                   }

                   if (x[0] > x[1]) {
                   int a, b;
                   a = x[0];
                   b = x[1];
                   x[0] = b;
                   x[1] = a;

                   a = y[0];
                   b = y[1];
                   y[0] = b;
                   y[1] = a;
                   } //(x[0] > x[1])

                  int delx = x[1]-x[0];
                  int dely = fabs(y[1]-y[0]);
                  int error = 0;
                  int x_n = x[0];
                  int y_n = y[0];
                  if (y[0] < y[1]) {ystep = 1;}
                  else {ystep = -1;}

                  for(n = 0; n<delx+1; n++) {
                       if (steep == 1) {
                        /*printf("[%i][%i][%u]\n", x_n, y_n, MASK[y_n*dimX+x_n]);*/
                        /*checks all points (classes) that the line crosses */
                        if (class_switcher == 0) {
                        if (MASK[j*dimX+i] != MASK[x_n*dimX+y_n]) {
                        	MASK_upd[x_n*dimX+y_n] = MASK[j*dimX+i];
                        	CORRECTEDRegions[x_n*dimX+y_n] += 1;
                       	 }}
                        if (class_switcher == 1) {
                          /* deal with  1 -> 2 -> 3 combination of classes, i.e. convert 2 (if exist) into 1 */
                           if (MASK[x_n*dimX+y_n] == ClassesList[2]) MASK_upd[x_n*dimX+y_n] = ClassesList[1];
                        }
                        if (class_switcher == 2) {
                          /* improbabale combination 0 -> 4 -> 3 (air->artifacts->liquor): 4 -> 3  or
                             improbabale (!) combination 0 -> 1 -> 3 (air->loop->liquor): 1 -> 3  */
                           if ((MASK[x_n*dimX+y_n] == ClassesList[4]) || (MASK[x_n*dimX+y_n] == ClassesList[1])) MASK_upd[x_n*dimX+y_n] = ClassesList[3];
                        }
                        if (class_switcher == 3) {
                          /* improbabale combination 0 -> 4 -> 1 (air->artifacts->loop): 4 -> 1 or
                             improbabale combination 0 -> 2 -> 1 (air->crystal->loop): 2 -> 1 or  */
                           if ((MASK[x_n*dimX+y_n] == ClassesList[4]) || (MASK[x_n*dimX+y_n] == ClassesList[2])) MASK_upd[x_n*dimX+y_n] = ClassesList[1];
                        }
                        if (class_switcher == 4) {
                          /* improbabale combination 0 -> 1 -> 2 (air->loop->crystal): 1 -> 2 */
                           if (MASK[x_n*dimX+y_n] == ClassesList[1]) MASK_upd[x_n*dimX+y_n] = ClassesList[2];
                        }
                      }
                       else {
                        // printf("[%i][%i][%u]\n", y_n, x_n, MASK[x_n*dimX+y_n]);
                        /*checks all points (classes) that the line crosses */
                        if (class_switcher == 0) {
                          if (MASK[j*dimX+i] != MASK[y_n*dimX+x_n]) {
	                           MASK_upd[y_n*dimX+x_n] = MASK[j*dimX+i];
                             CORRECTEDRegions[y_n*dimX+x_n] += 1;
                           }}
                        if (class_switcher == 1) {
                        /* deal with  1 -> 2 -> 3 combination of classes, i.e. convert 2 (if exist) into 1 */
                          if (MASK[y_n*dimX+x_n] == ClassesList[2]) MASK_upd[y_n*dimX+x_n] = ClassesList[1];
                        }
                        if (class_switcher == 2) {
                          /* improbabale combination 0 -> 4 -> 3 (air->artifacts->liquor): 4 -> 3  or
                             improbabale (!) combination 0 -> 1 -> 3 (air->loop->liquor): 1 -> 3  */
                           if ((MASK[x_n*dimX+y_n] == ClassesList[4]) || (MASK[x_n*dimX+y_n] == ClassesList[1])) MASK_upd[y_n*dimX+x_n] = ClassesList[3];
                        }
                        if (class_switcher == 3) {
                          /* improbabale combination 0 -> 4 -> 1 (air->artifacts->loop): 4 -> 1 or
                             improbabale combination 0 -> 2 -> 1 (air->crystal->loop): 2 -> 1 or  */
                           if ((MASK[y_n*dimX+x_n] == ClassesList[4]) || (MASK[y_n*dimX+x_n] == ClassesList[2])) MASK_upd[y_n*dimX+x_n] = ClassesList[1];
                        }
                        if (class_switcher == 4) {
                          /* improbabale combination 0 -> 1 -> 2 (air->loop->crystal): 1 -> 2 */
                           if (MASK[y_n*dimX+x_n] == ClassesList[1]) MASK_upd[y_n*dimX+x_n] = ClassesList[2];
                        }

                      }
                       x_n = x_n + 1;
                       error = error + dely;

                       if (2*error >= delx) {
                          y_n = y_n + ystep;
                         error = error - delx;
                       } // (2*error >= delx)
                       //printf("[%i][%i][%i]\n", X_new[n], Y_new[n], n) ;
                  } // for(int n = 0; n<delx+1; n++)
                  return 0;
}
/********************************************************************/
/***************************3D Functions*****************************/
/********************************************************************/
