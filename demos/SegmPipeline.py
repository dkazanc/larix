import h5py
import numpy as np
import matplotlib.pyplot as plt
import timeit
import geodesic_distance
from morphsnakes import morphological_chan_vese, circle_level_set
# using Morphological snakes from
# https://github.com/pmneila/morphsnakes

from ccpi.filters.regularisers import SB_TV
from ccpi.filters.regularisers import PatchSelect, NLTV

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk


h5f = h5py.File('/home/kjy41806/Documents/data_temp/TomoRec3D_13551.h5', 'r')
TomoRec3D_13551 = h5f['data'][:]
h5f.close()

#image = TomoRec3D_13551[100,10:None,:]
#image = TomoRec3D_13551[130,160:460,200:430] # 130-slice
# image = TomoRec3D_13551[130,:,:] # 130-slice
image = TomoRec3D_13551[40:140,:,:] 
image = image/np.max(image)

slices, NxSize, NySize = np.shape(image)
del TomoRec3D_13551
#%%
NLTV_im = np.float32(np.zeros(np.shape(image)))
print ("Doing NLTV denoising of X-Y slices")

for j in range(0,slices):
    # NLM processing of image
    pars = {'algorithm' : PatchSelect, \
            'input' : image[j,:,:],\
            'searchwindow': 9, \
            'patchwindow': 2,\
            'neighbours' : 20 ,\
            'edge_parameter':0.003}
    
    H_i, H_j, Weights = PatchSelect(pars['input'], 
                  pars['searchwindow'],
                  pars['patchwindow'], 
                  pars['neighbours'],
                  pars['edge_parameter'],'gpu')

    pars2 = {'algorithm' : NLTV, \
            'input' : image[j,:,:],\
            'H_i': H_i, \
            'H_j': H_j,\
            'H_k' : 0,\
            'Weights' : Weights,\
            'regularisation_parameter': 10.0,\
            'iterations': 100
            }
    NLTV_im[j,:,:] = NLTV(pars2['input'], 
                  pars2['H_i'],
                  pars2['H_j'], 
                  pars2['H_k'],
                  pars2['Weights'],
                  pars2['regularisation_parameter'],
                  pars2['iterations'])

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.imshow(NLTV_im[5,:,:], vmin=0.0, vmax=0.5, cmap="gray")
plt.title('NLTV denoised')
plt.show()
del H_i,H_j,Weights
#%%
print ("2D geodesic distance calculation for each slice")
slices, NxSize, NySize = np.shape(NLTV_im)
ls1 = np.uint8(np.zeros(np.shape(NLTV_im)))
for j in range(0,slices):
    ls1[j,:,:] = circle_level_set(tuple((NxSize, NySize)), (300, 300), 15)

segm3D_as2D = np.float32(np.zeros(np.shape(NLTV_im)))
for j in range(0,slices):
    segm3D_as2D[j,:,:] = geodesic_distance.geodesic2d_fast_marching(NLTV_im[j,:,:],ls1[j,:,:])

segm3D_as2D = segm3D_as2D/np.max(segm3D_as2D)

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('geodesic distance results')
plt.imshow(segm3D_as2D[5,:,:])
plt.pause(.2)
del NLTV_im
#%%
print ("Apply 3D TV denoising to the result of the Geodesic distance processing...")
pars = {'algorithm' : SB_TV, \
        'input' : segm3D_as2D,\
        'regularisation_parameter': 0.02, \
        'number_of_iterations' : 100 ,\
        'tolerance_constant': 0.0,\
        'methodTV': 0}

(SB_TV3D, info_vec_gpu) = SB_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],'gpu')

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.imshow(SB_TV3D[5,:,:], vmin=0.0, vmax=0.2, cmap="gray")
plt.title('SB_TV denoised')
plt.show()
plt.pause(.2)
del segm3D_as2D
#%%
print ("Running Morph Chan-Vese to get crystal...")
# image = SB_TV3D[35:45,:,:]
#ls1 = circle_level_set(SB_TV3D.shape, (5, 300, 300), 20)
start_time = timeit.default_timer()
# get the crystal 
CrystSegm = morphological_chan_vese(SB_TV3D, iterations=350, lambda1=1.0, lambda2=0.0035, init_level_set=ls1)
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

plt.figure()
plt.imshow(CrystSegm[5,:,:], vmin=0.0, vmax=1, cmap="gray")
plt.title('Crystal segmentation')
plt.show()
plt.pause(.2)
#%%
# morphologically process it
selem = disk(10)
CrystSegmMorph = np.uint8(np.zeros(np.shape(CrystSegm)))
slices, NxSize, NySize = np.shape(CrystSegm)

for j in range(0,slices):
    segm_to_proc = CrystSegm[j,:,:].copy()
    closing_t = closing(segm_to_proc, selem)
    segm_tmp = opening(closing_t, selem)
    CrystSegmMorph[j,:,:] = segm_tmp
#%
plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Morphological processing of crystal segmentation')
plt.imshow(CrystSegmMorph[5,:,:])
plt.pause(.2)
del CrystSegm
#%%
# now when crystal is segmented we can get the surrounding liquor
# initialise snakes with the result of crystal segmentation
print ("Running Morph Chan-Vese to get the surrounding liquor...")
start_time = timeit.default_timer()
# get the crystal 
LiquorSegm = morphological_chan_vese(SB_TV3D, iterations=350, smoothing=1, lambda1=1.0, lambda2=0.045, init_level_set=CrystSegmMorph)
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

plt.figure()
plt.imshow(LiquorSegm[5,:,:], vmin=0.0, vmax=1, cmap="gray")
plt.title('Liquor segmentation')
plt.pause(.2)
plt.show()
#%%
# morphologically process it
selem = disk(10)
LiquorSegmMorph = np.uint8(np.zeros(np.shape(LiquorSegm)))
slices, NxSize, NySize = np.shape(LiquorSegm)

for j in range(0,slices):
    segm_to_proc = LiquorSegm[j,:,:].copy()
    closing_t = closing(segm_to_proc, selem)
    segm_tmp = opening(closing_t, selem)
    LiquorSegmMorph[j,:,:] = segm_tmp
#%
plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Morphological processing of liquor segmentation')
plt.imshow(LiquorSegmMorph[5,:,:])
plt.pause(.2)
del LiquorSegm
#%%
# getting the whole object
print ("Running Morph Chan-Vese to get the whole object...")
dist=5
init_set = np.uint8(np.zeros(np.shape(image)))
init_set[0:slices, dist:NxSize-dist, dist:NySize-dist] = 1

start_time = timeit.default_timer()
# get the crystal 
WholeObjSegm = morphological_chan_vese(SB_TV3D, iterations=350, smoothing=1, lambda1=0.1, lambda2=1.0, init_level_set=init_set)
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('The whole object segmentation')
plt.imshow(WholeObjSegm[5,:,:])
plt.pause(.2)
#%%
# morphologically process it
selem = disk(10)
WholeObjSegmMorph = np.uint8(np.zeros(np.shape(WholeObjSegm)))
slices, NxSize, NySize = np.shape(WholeObjSegm)

for j in range(0,slices):
    segm_to_proc = WholeObjSegm[j,:,:].copy()
    closing_t = closing(segm_to_proc, selem)
    segm_tmp = opening(closing_t, selem)
    WholeObjSegmMorph[j,:,:] = segm_tmp
#%
plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Morphological processing of the whole object segmentation')
plt.imshow(WholeObjSegmMorph[5,:,:])
plt.pause(.2)
del WholeObjSegm
#%%
# combining everything into a single object - crystal, liquor, loop, vacuum
# 
FinalSegm = CrystSegmMorph+LiquorSegmMorph+WholeObjSegmMorph

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Final segmentation')
plt.imshow(FinalSegm[5,:,:])
plt.pause(.2)
#%%