import h5py
import numpy as np
import matplotlib.pyplot as plt
import timeit

import geodesic_distance

# using Morphological snakes from
# https://github.com/pmneila/morphsnakes
from morphsnakes import morphological_chan_vese, circle_level_set


from ccpi.filters.regularisers import SB_TV
from ccpi.filters.regularisers import PatchSelect, NLTV

from skimage.morphology import opening, closing
from skimage.morphology import disk

def initialiseLS(slices, NxSize, NySize, coordX0, coordY0, coordZ0, coordX1, coordY1, coordZ1, circle_size):
    LS_init = np.uint8(np.zeros((slices, NxSize, NySize)))
    # calculate coordinates
    steps = coordZ1 - coordZ0
    if ((steps <= 0) or (steps > slices)):
        raise Exception("Z coordinates are given incorrectly (out of array range)")
    distance = np.sqrt((coordX1 - coordX0)**2 + (coordY1 - coordY0)**2)
    d_dist = distance/(steps-1)
    d_step = d_dist
    
    for j in range(coordZ0,coordZ1):
        t = d_step/distance
        x_t = np.round((1.0 - t)*coordX0 + t*coordX1)
        y_t = np.round((1.0 - t)*coordY0 + t*coordY1)
        if (coordX0 == coordX1):
            x_t = coordX0
        if(coordY0 == coordY1):
            y_t = coordY0
        LS_init[j,:,:] = circle_level_set(tuple((NxSize, NySize)), (y_t, x_t), circle_size)
        d_step += d_dist
    return LS_init

def morphological_proc(data, disk_size):
    selem = disk(disk_size)
    Morph = np.uint8(np.zeros(np.shape(data)))
    slices, NxSize, NySize = np.shape(data)
    for j in range(0,slices):
        segm_to_proc = data[j,:,:].copy()
        closing_t = closing(segm_to_proc, selem)
        segm_tmp = opening(closing_t, selem)
        Morph[j,:,:] = segm_tmp
    return Morph

h5f = h5py.File('/scratch/data_temp/i23/TomoRec3D_13551.h5', 'r')
TomoRec3D_13551 = h5f['data'][:]
h5f.close()

# image = TomoRec3D_13551[130,:,:] # 130-slice
image = TomoRec3D_13551[170:190,:,:]
#image = TomoRec3D_13551[180:190,:,:]
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
            'iterations': 50
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
print ("Geodesic distance calculation")
slices, NxSize, NySize = np.shape(NLTV_im)
#initialise level sets
#ls1 = initialiseLS(slices, NxSize, NySize, coordX0=260, coordY0=243, coordZ0=0,\
#                   coordX1=362, coordY1=373, coordZ1=slices, circle_size = 10)

ls1 = initialiseLS(slices, NxSize, NySize, coordX0=360, coordY0=360, coordZ0=0,\
                   coordX1=360, coordY1=360, coordZ1=slices, circle_size = 10)


segm3D_geo = np.float32(np.zeros(np.shape(NLTV_im)))
for j in range(0,slices):
    #segm3D_geo[j,:,:] = geodesic_distance.geodesic2d_fast_marching(NLTV_im[j,:,:],ls1[j,:,:])
    #using raster scan (faster)
    segm3D_geo[j,:,:] = geodesic_distance.geodesic2d_raster_scan(NLTV_im[j,:,:], ls1[j,:,:], 0.5, 4)


"""
start_time = timeit.default_timer()
segm3D_geo = geodesic_distance.geodesic3d_raster_scan(NLTV_im,ls1, 0.5, 4)
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
"""
segm3D_geo = segm3D_geo/np.max(segm3D_geo)

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('geodesic distance results')
plt.imshow(segm3D_geo[5,:,:], vmin=0.0, vmax=0.13, cmap="gray")
plt.pause(.2)
#del NLTV_im
#%%
print ("Apply 3D TV denoising to the result of the Geodesic distance processing...")
pars = {'algorithm' : SB_TV, \
        'input' : segm3D_geo,\
        'regularisation_parameter': 0.033, \
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
plt.imshow(SB_TV3D[5,:,:], vmin=0.0, vmax=0.13, cmap="gray")
plt.title('SB_TV denoised')
plt.show()
plt.pause(.2)

np.save('GeoDistanceTV13551_200slices.npy', SB_TV3D)
#del segm3D_geo
#%%

print ("Running Morph Chan-Vese (3D) to get crystal...")
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
CrystSegmMorph = morphological_proc(data = CrystSegm, disk_size=7)

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Morphological processing of crystal segmentation')
plt.imshow(CrystSegmMorph[5,:,:])
plt.pause(.2)
#del CrystSegm
#%%
# now when crystal is segmented we can get the surrounding liquor
# initialise snakes with the result of crystal segmentation
print ("Running Morph Chan-Vese (3D) to get the surrounding liquor...")
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
LiquorSegmMorph = morphological_proc(data = LiquorSegm, disk_size=7)

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Morphological processing of liquor segmentation')
plt.imshow(LiquorSegmMorph[5,:,:])
plt.pause(.2)
#del LiquorSegm
#%%
# getting the whole object
print ("Running Morph Chan-Vese (3D) to get the whole object...")
# initialise level sets
#dist=1
#init_set = np.uint8(np.zeros(np.shape(image)))
#init_set[0:slices, dist:NxSize-dist, dist:NySize-dist] = 1

start_time = timeit.default_timer()
# get the crystal 
WholeObjSegm = morphological_chan_vese(SB_TV3D, iterations=350, smoothing=1, lambda1=0.48, lambda2=1.0, init_level_set=LiquorSegmMorph)
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('The whole object segmentation')
plt.imshow(WholeObjSegm[5,:,:])
plt.pause(.2)
#%%
# morphologically process it
WholeObjSegmMorph = morphological_proc(data = WholeObjSegm, disk_size=7)

#%
plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Morphological processing of the whole object segmentation')
plt.imshow(WholeObjSegmMorph[5,:,:])
plt.pause(.2)
del WholeObjSegm
#%%
# combining everything into a single object - crystal, liquor, loop, vacuum
FinalSegm = CrystSegmMorph+LiquorSegmMorph+WholeObjSegmMorph

for j in range(0,slices):
    mask_temp = FinalSegm[j,:,:]
    res = np.where(mask_temp == 0)
    if (np.size(res[0]) == 0 & np.size(res[1]) == 0):
        mask_temp[mask_temp == 1] = 0
        FinalSegm[j,:,:] = mask_temp

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Final segmentation')
plt.imshow(FinalSegm[5,:,:])
plt.pause(.2)
np.save('FinalSegm13551_2_200slices.npy', FinalSegm)
#%%