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
 * Input Parameters (from Python):
 * 1. MASK [0:255], the result of some classification algorithm (information-based preferably, Gaussian Mixtures works quite well)
 * 2. The list of classes needs to be processed. The order matters, e.g. (air, crystal)
 * 3. The list of improbable combinations of classes, such as: (class_start, class_middle, class_end, class_substiture)
 * 4. The size of the Correction Window (neighbourhood window)
 * 5. The number of iterations

 * Output:
 * 1. MASK_upd - the UPDATED MASK where some regions have been corrected (merged) or removed
 * 2. CORRECTEDRegions - The array of the same size as MASK where all regions which were
 * changed are highlighted and the changes have been counted
 */

int signum(int i) {    
    return (i>0)?1:((i<0)?-1:0);    
}

#ifndef max
    #define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif
 
#ifndef min
    #define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif


float Mask_merge_main(unsigned char *MASK, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *SelClassesList, unsigned char *ComboClasses, int tot_combinations, int SelClassesList_length, int classesNumb, int CorrectionWindow, int iterationsNumb, int dimX, int dimY, int dimZ)
{
    long i,j,k,l,n;
    int counterG, switcher;
    long DimTotal;
    unsigned char *MASK_temp, *ClassesList, CurrClass, temp, class_start, class_mid, class_end, class_substitute;
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
     /********************** PERFORM 2D MASK PROCESSING ************************/
    /* start iterations */
    for(k=0; k<iterationsNumb; k++) {   
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
        Mask_update_main2D(MASK_temp, MASK_upd, CORRECTEDRegions, i, j, CorrectionWindow, (long)(dimX), (long)(dimY));
       	  }}
      }}
      /* copy the updated mask */
      copyIm_unchar(MASK_upd, MASK_temp, (long)(dimX), (long)(dimY), (long)(dimZ));     
      }
       /* Main classes have been processed. Working with implausable combinations */
       /* loop over the combinations of 3 */      
       for(l=0; l<tot_combinations; l++) {
	 class_start = ComboClasses[l*4]; /* current class */
	 class_mid = ComboClasses[l*4+1]; /* class in-between */
	 class_end = ComboClasses[l*4+2]; /* neighbour class */
	 class_substitute = ComboClasses[l*4+3]; /* class to replace class_mid with */
	 /*printf("[%i][%u][%u][%u][%u]\n", l, class_start, class_mid, class_end, class_substitute);*/
	 #pragma omp parallel for shared(MASK_temp,MASK_upd, l, class_start, class_mid, class_end, class_substitute) private(i,j)
	     for(i=0; i<dimX; i++) {
	             for(j=0; j<dimY; j++) { 
	        Mask_update_combo2D(MASK_temp, MASK_upd, CORRECTEDRegions, ClassesList, class_start, class_mid, class_end, class_substitute, i, j, CorrectionWindow, (long)(dimX), (long)(dimY));        
		}}		        
         copyIm_unchar(MASK_upd, MASK_temp, (long)(dimX), (long)(dimY), (long)(dimZ));          
       }             
      }
    }
    else {
    /********************** PERFORM 3D MASK PROCESSING ************************/
    /* start iterations */
    for(l=0; l<iterationsNumb; l++) {   
    #pragma omp parallel for shared(MASK,MASK_upd) private(i,j,k)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
           for(k=0; k<dimZ; k++) {
    /* STEP1: in a smaller neighbourhood check that the current pixel is NOT an outlier */
    OutiersRemoval3D(MASK, MASK_upd, i, j, k, (long)(dimX), (long)(dimY), (long)(dimZ));
		    }}}
    /* copy the updated MASK (clean of outliers) */
    copyIm_unchar(MASK_upd, MASK_temp, (long)(dimX), (long)(dimY), (long)(dimZ));

    // printf("[%u][%u][%u]\n", ClassesList[0], ClassesList[1], ClassesList[2]);
    for(n=0; n<SelClassesList_length; n++) {
    /*printf("[%u]\n", ClassesList[SelClassesList[l]]);*/
    #pragma omp parallel for shared(MASK_temp,MASK_upd,l,n) private(i,j,k)
    for(i=0; i<dimX; i++) {
        for(j=0; j<dimY; j++) {
           for(k=0; k<dimZ; k++) {
      /* The class of the central pixel has not changed, i.e. the central pixel is not an outlier -> continue */
      if (MASK_temp[(dimX*dimY)*k + j*dimX+i] == MASK[(dimX*dimY)*k + j*dimX+i]) {
	/* !One needs to work with a specific class to avoid overlaps! It is
	     crucial to establish relevant classes first (given as an input in SelClassesList) */
     if (MASK_temp[(dimX*dimY)*k + j*dimX+i] == ClassesList[SelClassesList[n]]) {
        Mask_update_main3D(MASK_temp, MASK_upd, CORRECTEDRegions, i, j, k, CorrectionWindow, (long)(dimX), (long)(dimY), (long)(dimZ));
       	  }}
      }}}
      /* copy the updated mask */
      copyIm_unchar(MASK_upd, MASK_temp, (long)(dimX), (long)(dimY), (long)(dimZ));     
    }    
    /* Main classes have been processed. Working with implausable combinations */
    /* loop over the combinations of 3 */      
       for(n=0; n<tot_combinations; n++) {
	 class_start = ComboClasses[n*4]; /* current class */
	 class_mid = ComboClasses[n*4+1]; /* class in-between */
	 class_end = ComboClasses[n*4+2]; /* neighbour class */
	 class_substitute = ComboClasses[n*4+3]; /* class to replace class_mid with */
	 /*printf("[%i][%u][%u][%u][%u]\n", l, class_start, class_mid, class_end, class_substitute);*/
	 #pragma omp parallel for shared(MASK_temp,MASK_upd, n, l, class_start, class_mid, class_end, class_substitute) private(i,j,k)
	     for(i=0; i<dimX; i++) {
                  for(j=0; j<dimY; j++) { 
	              for(k=0; k<dimZ; k++) {
	        Mask_update_combo3D(MASK_temp, MASK_upd, CORRECTEDRegions, ClassesList, class_start, class_mid, class_end, class_substitute, i, j, k, CorrectionWindow, (long)(dimX), (long)(dimY), (long)(dimZ));        
		}}}		        
         copyIm_unchar(MASK_upd, MASK_temp, (long)(dimX), (long)(dimY), (long)(dimZ));          
       }    
	    } /* iterations terminated*/      
    }    
    free(MASK_temp);
    free(ClassesList);
    return *MASK_upd;
}


float mask_merge_binary_main(unsigned char *MASK, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, int selectedClass, int CorrectionWindow, int iterationsNumb, int dimX, int dimY, int dimZ)
{
    long i,j,k,l;
    long DimTotal;
    unsigned char *MASK_temp;
    DimTotal = (long)(dimX*dimY*dimZ);

    MASK_temp = (unsigned char*) calloc (DimTotal,sizeof(unsigned char));

    /* copy given MASK to MASK_upd*/
    copyIm_unchar(MASK, MASK_upd, (long)(dimX), (long)(dimY), (long)(dimZ));
	
    if (dimZ == 1) {
     /********************** PERFORM 2D MASK PROCESSING ************************/
    /* start iterations */
    for(k=0; k<iterationsNumb; k++) {   
    #pragma omp parallel for shared(MASK,MASK_upd) private(i,j)
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {
    /* STEP1: in a smaller neighbourhood check that the current pixel is NOT an outlier */
    OutiersRemoval2D(MASK, MASK_upd, i, j, (long)(dimX), (long)(dimY));
        }}
    /* copy the updated MASK (clean of outliers) to MASK_temp*/
    copyIm_unchar(MASK_upd, MASK_temp, (long)(dimX), (long)(dimY), (long)(dimZ));
    
    #pragma omp parallel for shared(MASK_temp,MASK_upd) private(i,j)
    for(j=0; j<dimY; j++) {
        for(i=0; i<dimX; i++) {        
      /* The class of the central pixel has not changed, i.e. the central pixel is not an outlier -> continue */
      if (MASK_temp[j*dimX+i] == MASK[j*dimX+i]) {
    	/* !One needs to work with a specific class to avoid overlaps */
       if (MASK_temp[j*dimX+i] == selectedClass) {
        Mask_update_main2D(MASK_temp, MASK_upd, CORRECTEDRegions, i, j, CorrectionWindow, (long)(dimX), (long)(dimY));
       	  }}
      }}
      /* copy the updated mask */
      copyIm_unchar(MASK_upd, MASK_temp, (long)(dimX), (long)(dimY), (long)(dimZ));          
      }
    }
    else {
    /********************** PERFORM 3D MASK PROCESSING ************************/
    /* start iterations */
    for(l=0; l<iterationsNumb; l++) {   
    #pragma omp parallel for shared(MASK,MASK_upd) private(i,j,k)
    for(k=0; k<dimZ; k++) {
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
    /* STEP1: in a smaller neighbourhood check that the current pixel is NOT an outlier */
    OutiersRemoval3D(MASK, MASK_upd, i, j, k, (long)(dimX), (long)(dimY), (long)(dimZ));
    	}}}
    /* copy the updated MASK (clean of outliers) */
    copyIm_unchar(MASK_upd, MASK_temp, (long)(dimX), (long)(dimY), (long)(dimZ));
        
    #pragma omp parallel for shared(MASK_temp,MASK_upd,l) private(i,j,k)
    for(k=0; k<dimZ; k++) {
        for(j=0; j<dimY; j++) {
            for(i=0; i<dimX; i++) {
    /* The class of the central pixel has not changed, i.e. the central pixel is not an outlier -> continue */
      if (MASK_temp[(dimX*dimY)*k + j*dimX+i] == MASK[(dimX*dimY)*k + j*dimX+i]) {
	/* !One needs to work with a specific class to avoid overlaps */
     if (MASK_temp[(dimX*dimY)*k + j*dimX+i] == selectedClass) {
        Mask_update_main3D(MASK_temp, MASK_upd, CORRECTEDRegions, i, j, k, CorrectionWindow, (long)(dimX), (long)(dimY), (long)(dimZ));
       	  }}
      }}}
      /* copy the updated mask */
      copyIm_unchar(MASK_upd, MASK_temp, (long)(dimX), (long)(dimY), (long)(dimZ));      
      } /* iterations terminated*/      
    }   
    free(MASK_temp);
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

float Mask_update_main2D(unsigned char *MASK_temp, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, long i, long j, int CorrectionWindow, long dimX, long dimY)
{
  long i_m, j_m, i1, j1, CounterOtherClass;

  /* STEP2: in a larger neighbourhood check that the other class is present in the neighbourhood */
  CounterOtherClass = 0;
  for(i_m=-CorrectionWindow; i_m<=CorrectionWindow; i_m++) {
      for(j_m=-CorrectionWindow; j_m<=CorrectionWindow; j_m++) {
        i1 = i+i_m;
        j1 = j+j_m;
        if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {
          if (MASK_temp[j*dimX+i] != MASK_temp[j1*dimX+i1]) CounterOtherClass++;
        }
        if (CounterOtherClass > 0) break;
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
              bresenham2D_main(i, j, i1, j1, MASK_temp, MASK_upd, CORRECTEDRegions, (long)(dimX), (long)(dimY));
             }
            }
          }}
      }
  return *MASK_upd;
}

int bresenham2D_main(int i, int j, int i1, int j1, unsigned char *MASK, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, long dimX, long dimY)
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
                       /* this replaces any class which is different from classes in
                       the starting and ending points */
                        if (MASK[j*dimX+i] != MASK[x_n*dimX+y_n]) {
                        	MASK_upd[x_n*dimX+y_n] = MASK[j*dimX+i];
                        	CORRECTEDRegions[x_n*dimX+y_n] += 1;
                       	 }                     
                      }
                       else {                       
                          if (MASK[j*dimX+i] != MASK[y_n*dimX+x_n]) {
	                     MASK_upd[y_n*dimX+x_n] = MASK[j*dimX+i];
                             CORRECTEDRegions[y_n*dimX+x_n] += 1;
                           }                      
                      }
                       x_n = x_n + 1;
                       error = error + dely;
                       if (2*error >= delx) {
                          y_n = y_n + ystep;
                         error = error - delx;
                       } // (2*error >= delx)                       
                  } // for(int n = 0; n<delx+1; n++)
                  return 0;
}

float Mask_update_combo2D(unsigned char *MASK_temp, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *ClassesList, unsigned char class_start, unsigned char class_mid, unsigned char class_end, unsigned char class_substitute, long i, long j, int CorrectionWindow, long dimX, long dimY)
{
  long i_m, j_m, i1, j1, CounterOtherClass;

  /* STEP2: in a larger neighbourhood check that the other class is present in the neighbourhood  */
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
      STEP 3: Loop through all neighbours in CorrectionWindow and check the spatial connections.
      Check that if there are any classes between points A and B that does not belong to A and B (A,B \in C)
      */
      for(i_m=-CorrectionWindow; i_m<=CorrectionWindow; i_m++) {
          for(j_m=-CorrectionWindow; j_m<=CorrectionWindow; j_m++) {
            i1 = i+i_m;
            j1 = j+j_m;
            if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY))) {           
              if ((MASK_temp[j*dimX+i] == ClassesList[class_start]) && (MASK_temp[j1*dimX+i1] == ClassesList[class_end])) {
              /* We check that point A belongs to "class_start" and point B to "class_end". If they do then the idea is to check if 
              "class_mid" (undesirable class) lies inbetween two classes. If it does -> replace it with "class_substitute".  */
              bresenham2D_combo(i, j, i1, j1, MASK_temp, MASK_upd, CORRECTEDRegions, ClassesList, class_mid, class_substitute, (long)(dimX), (long)(dimY));
              }              
            }
          }}
      }
  return *MASK_upd;
}

int bresenham2D_combo(int i, int j, int i1, int j1, unsigned char *MASK, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *ClassesList,  unsigned char class_mid, unsigned char class_substitute, long dimX, long dimY)
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
                        /* dealing with various improbable combination of classes in the mask. The improbable class is replaced 
                        with more probable one. */
                        if (MASK[x_n*dimX+y_n] == ClassesList[class_mid]) {                        
	                        MASK_upd[x_n*dimX+y_n] = ClassesList[class_substitute];
	                        CORRECTEDRegions[x_n*dimX+y_n] += 1;
	                        }
                      }
                       else {
                        // printf("[%i][%i][%u]\n", y_n, x_n, MASK[x_n*dimX+y_n]);
                        /* dealing with various improbable combination of classes in the mask. The improbable class is replaced 
                        with more probable one. */
                          if (MASK[y_n*dimX+x_n] == ClassesList[class_mid]) {                          
	                          MASK_upd[y_n*dimX+x_n] = ClassesList[class_substitute];
	                          CORRECTEDRegions[y_n*dimX+x_n] += 1;
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
float OutiersRemoval3D(unsigned char *MASK, unsigned char *MASK_upd, long i, long j, long k, long dimX, long dimY, long dimZ)
{
  /*if the ROI pixel does not belong to the surrondings, turn it into the surronding*/
  long i_m, j_m, k_m, i1, j1, k1, counter;
    counter = 0;
    for(i_m=-1; i_m<=1; i_m++) {
      for(j_m=-1; j_m<=1; j_m++) {
        for(k_m=-1; k_m<=1; k_m++) {
        i1 = i+i_m;
        j1 = j+j_m;
        k1 = k+k_m;
        if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY)) && ((k1 >= 0) && (k1 < dimZ))) {
          if (MASK[(dimX*dimY)*k + j*dimX+i] != MASK[(dimX*dimY)*k1 + j1*dimX+i1]) counter++;
        }
      }}}
      if (counter >= 25) MASK_upd[(dimX*dimY)*k + j*dimX+i] = MASK[(dimX*dimY)*k1 + j1*dimX+i1];
      return *MASK_upd;
}

float Mask_update_main3D(unsigned char *MASK_temp, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, long i, long j, long k, int CorrectionWindow, long dimX, long dimY, long dimZ)
{
  long i_m, j_m, k_m, i1, j1, k1, CounterOtherClass;

  /* STEP2: in a larger neighbourhood check first that the other class is present in the neighbourhood */
  CounterOtherClass = 0;
  for(i_m=-CorrectionWindow; i_m<=CorrectionWindow; i_m++) {
      for(j_m=-CorrectionWindow; j_m<=CorrectionWindow; j_m++) {
          for(k_m=-CorrectionWindow; k_m<=CorrectionWindow; k_m++) {
	        i1 = i+i_m;
        	j1 = j+j_m;
        	k1 = k+k_m;
        if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY)) && ((k1 >= 0) && (k1 < dimZ))) {
          if (MASK_temp[(dimX*dimY)*k + j*dimX+i] != MASK_temp[(dimX*dimY)*k1 + j1*dimX+i1]) CounterOtherClass++;
        }
         if (CounterOtherClass > 0) break;
      }}}
      if (CounterOtherClass > 0) {
      /* the other class is present in the vicinity of CorrectionWindow, continue to STEP 3 */
      /*
      STEP 3: Loop through all neighbours in CorrectionWindow and check the spatial connection.
      Meaning that we're instrested if there are any classes between points A and B that
      does not belong to A and B (A,B \in C)
      */
      for(i_m=-CorrectionWindow; i_m<=CorrectionWindow; i_m++) {
          for(j_m=-CorrectionWindow; j_m<=CorrectionWindow; j_m++) {
	      for(k_m=-CorrectionWindow; k_m<=CorrectionWindow; k_m++) {
	        i1 = i+i_m;
        	j1 = j+j_m;
        	k1 = k+k_m;
        if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY)) && ((k1 >= 0) && (k1 < dimZ))) {
              if (MASK_temp[(dimX*dimY)*k + j*dimX+i] == MASK_temp[(dimX*dimY)*k1 + j1*dimX+i1]) {
               /* A and B points belong to the same class, do STEP 4*/
               /* STEP 4: Run the Bresenham line algorithm between A and B points
               and convert all points on the way to the class of A. */
              bresenham3D_main(i, j, k, i1, j1, k1, MASK_temp, MASK_upd, CORRECTEDRegions, (long)(dimX), (long)(dimY), (long)(dimZ));
             }
            }
          }}}
      }
  return *MASK_upd;
}

int bresenham3D_main(int i, int j, int k, int i1, int j1, int k1, unsigned char *MASK, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, long dimX, long dimY, long dimZ)
{
    int P1[] = {i, j, k};
    int P2[] = {i1, j1, k1};
    
    int x1 = P1[0];
    int y1 = P1[1];
    int z1 = P1[2];
     
    int x2 = P2[0];
    int y2 = P2[1];
    int z2 = P2[2];
     
    int dx = x2 - x1;
    int dy = y2 - y1;
    int dz = z2 - z1;
     
    int ax = fabs(dx)*2;
    int ay = fabs(dy)*2;
    int az = fabs(dz)*2;
     
    int sx = signum(dx);
    int sy = signum(dy);
    int sz = signum(dz);
     
    int x = x1;
    int y = y1;
    int z = z1;
     
    int xd;
    int yd;
    int zd;
    
    //printf("ijk indeces: [%i][%i][%i]\n", i, j, k) ;
     
    if (ax >= max(ay, az)) {
        int yd = ay - ax/2;
        int zd = az - ax/2;
         
        while (1) {             
            // printf("xyz indeces: [%i][%i][%i]\n", x, y, z) ;
            
            if (MASK[(dimX*dimY)*k + j*dimX+i] != MASK[(dimX*dimY)*z + y*dimX+x]) {
                       	MASK_upd[(dimX*dimY)*z + y*dimX+x] = MASK[(dimX*dimY)*k + j*dimX+i];
                       	CORRECTEDRegions[(dimX*dimY)*z + y*dimX+x] += 1;
                       	 }       
                         
            if (x == x2)  break;
             
            if (yd >= 0) {
                y = y + sy;     // move along y
                yd = yd - ax; }
             
            if (zd >= 0) {
                z = z + sz; // % move along z
                zd = zd - ax;
            }
             
            x  = x  + sx;       // move along x
            yd = yd + ay;
            zd = zd + az;
        } //while
         
    } // (ax>= fmax(ay,az))
    else if (ay >= max(ax, az)) {
        xd = ax - ay/2;
        zd = az - ay/2;
         
        while (1) {             

            if (MASK[(dimX*dimY)*k + j*dimX+i] != MASK[(dimX*dimY)*z + y*dimX+x]) {
                     	MASK_upd[(dimX*dimY)*z + y*dimX+x] = MASK[(dimX*dimY)*k + j*dimX+i];
                      	CORRECTEDRegions[(dimX*dimY)*z + y*dimX+x] += 1;
                    	 }
             
            if (y == y2)  break;
             
            if (xd >= 0) {
                x = x + sx;     // move along x
                xd = xd - ay;
            }
             
            if (zd >= 0)  {
                z = z + sz; //move along z
                zd = zd - ay;
            }
             
            y  = y  + sy;       // % move along y
            xd = xd + ax;
            zd = zd + az;
        } // while
    }
    else if (az >= max(ax, ay)) {
        xd = ax - az/2;
        yd = ay - az/2;
         
        while (1) {

            if (MASK[(dimX*dimY)*k + j*dimX+i] != MASK[(dimX*dimY)*z + y*dimX+x]) {
                       	MASK_upd[(dimX*dimY)*z + y*dimX+x] = MASK[(dimX*dimY)*k + j*dimX+i];
                       	CORRECTEDRegions[(dimX*dimY)*z + y*dimX+x] += 1;
             } 
             
            if(z == z2)  break;
             
            if(xd >= 0)  {
                x = x + sx; // move along x
                xd = xd - az;
            }             
            if (yd >= 0) {
                y = y + sy; // % move along y
                yd = yd - az;
            }
             
            z  = z  + sz;       //% move along z
            xd = xd + ax;
            yd = yd + ay;
             
        } //while loop
    }
 return 0;
}

float Mask_update_combo3D(unsigned char *MASK_temp, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *ClassesList, unsigned char class_start, unsigned char class_mid, unsigned char class_end, unsigned char class_substitute, long i, long j, long k, int CorrectionWindow, long dimX, long dimY, long dimZ)
{
  long i_m, j_m, k_m, i1, j1, k1, CounterOtherClass;

  /* STEP2: in a larger neighbourhood check that the other class is present in the neighbourhood  */
  CounterOtherClass = 0;
  for(i_m=-CorrectionWindow; i_m<=CorrectionWindow; i_m++) {
      for(j_m=-CorrectionWindow; j_m<=CorrectionWindow; j_m++) {
          for(k_m=-CorrectionWindow; k_m<=CorrectionWindow; k_m++) {
	        i1 = i+i_m;
        	j1 = j+j_m;
        	k1 = k+k_m;
        if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY)) && ((k1 >= 0) && (k1 < dimZ))) {
          if (MASK_temp[(dimX*dimY)*k + j*dimX+i] != MASK_temp[(dimX*dimY)*k1 + j1*dimX+i1]) CounterOtherClass++;
        }
         if (CounterOtherClass > 0) break;
      }}}
      if (CounterOtherClass > 0) {
      /* the other class is present in the vicinity of CorrectionWindow, continue to STEP 3 */
      /*
      STEP 3: Loop through all neighbours in CorrectionWindow and check the spatial connections.
      Check that if there are any classes between points A and B that does not belong to A and B (A,B \in C)
      */
      for(i_m=-CorrectionWindow; i_m<=CorrectionWindow; i_m++) {
          for(j_m=-CorrectionWindow; j_m<=CorrectionWindow; j_m++) {
    	      for(k_m=-CorrectionWindow; k_m<=CorrectionWindow; k_m++) {
	        i1 = i+i_m;
        	j1 = j+j_m;
        	k1 = k+k_m;
        if (((i1 >= 0) && (i1 < dimX)) && ((j1 >= 0) && (j1 < dimY)) && ((k1 >= 0) && (k1 < dimZ))) {      
              if ((MASK_temp[(dimX*dimY)*k + j*dimX+i] == ClassesList[class_start]) && (MASK_temp[(dimX*dimY)*k1 + j1*dimX+i1] == ClassesList[class_end])) {
              /* We check that point A belongs to "class_start" and point B to "class_end". If they do then the idea is to check if 
              "class_mid" (undesirable class) lies inbetween two classes. If it does -> replace it with "class_substitute".  */
              bresenham3D_combo(i, j, k, i1, j1, k1, MASK_temp, MASK_upd, CORRECTEDRegions, ClassesList, class_mid, class_substitute, (long)(dimX), (long)(dimY), (long)(dimZ));
              }              
            }
          }}}
      }
  return *MASK_upd;
}

int bresenham3D_combo(int i, int j, int k, int i1, int j1, int k1, unsigned char *MASK, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *ClassesList,  unsigned char class_mid, unsigned char class_substitute, long dimX, long dimY, long dimZ)
{
    int P1[] = {i, j, k};
    int P2[] = {i1, j1, k1};

    int x1 = P1[0];
    int y1 = P1[1];
    int z1 = P1[2];
     
    int x2 = P2[0];
    int y2 = P2[1];
    int z2 = P2[2];
     
    int dx = x2 - x1;
    int dy = y2 - y1;
    int dz = z2 - z1;
     
    int ax = fabs(dx)*2;
    int ay = fabs(dy)*2;
    int az = fabs(dz)*2;
     
    int sx = signum(dx);
    int sy = signum(dy);
    int sz = signum(dz);
     
    int x = x1;
    int y = y1;
    int z = z1;
     
    int xd;
    int yd;
    int zd;
    
    //printf("ijk indeces: [%i][%i][%i]\n", i, j, k) ;
     
    if (ax >= max(ay, az)) {
        int yd = ay - ax/2;
        int zd = az - ax/2;
         
        while (1) {
             
            //fprintf(stderr,"\nid: %d ",idx);
            /* getting the indeces of voxels which were crossed by the line */
                                   	        
           if (MASK[(dimX*dimY)*z + y*dimX+x] == ClassesList[class_mid]) {                        
	                        MASK_upd[(dimX*dimY)*z + y*dimX+x] = ClassesList[class_substitute];
	                        CORRECTEDRegions[(dimX*dimY)*z + y*dimX+x] += 1;
	                        }
                         
            if (x == x2)  break;
             
            if (yd >= 0) {
                y = y + sy;     // move along y
                yd = yd - ax; }
             
            if (zd >= 0) {
                z = z + sz; // % move along z
                zd = zd - ax;
            }
             
            x  = x  + sx;       // move along x
            yd = yd + ay;
            zd = zd + az;
        } //while
         
    } // (ax>= fmax(ay,az))
    else if (ay >= max(ax, az)) {
        xd = ax - ay/2;
        zd = az - ay/2;
         
        while (1) {            

           if (MASK[(dimX*dimY)*z + y*dimX+x] == ClassesList[class_mid]) {                        
	                        MASK_upd[(dimX*dimY)*z + y*dimX+x] = ClassesList[class_substitute];
	                        CORRECTEDRegions[(dimX*dimY)*z + y*dimX+x] += 1;
	                        }
             
            if (y == y2)  break;
             
            if (xd >= 0) {
                x = x + sx;     // move along x
                xd = xd - ay;
            }
             
            if (zd >= 0)  {
                z = z + sz; //move along z
                zd = zd - ay;
            }
             
            y  = y  + sy;       // % move along y
            xd = xd + ax;
            zd = zd + az;
        } // while
    }
    else if (az >= max(ax, ay)) {
        xd = ax - az/2;
        yd = ay - az/2;
         
        while (1) {

           if (MASK[(dimX*dimY)*z + y*dimX+x] == ClassesList[class_mid]) {                        
	                        MASK_upd[(dimX*dimY)*z + y*dimX+x] = ClassesList[class_substitute];
	                        CORRECTEDRegions[(dimX*dimY)*z + y*dimX+x] += 1;
	                        }
             
            if(z == z2)  break;
             
            if(xd >= 0)  {
                x = x + sx; // move along x
                xd = xd - az;
            }             
            if (yd >= 0) {
                y = y + sy; // % move along y
                yd = yd - az;
            }
             
            z  = z  + sz;   //% move along z
            xd = xd + ax;
            yd = yd + ay;
             
        } //while loop
    }
 return 0;
}


