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
#%%
h5f = h5py.File('/dls/i12/data/2019/nt23252-1/tmp/daniil_tmp/data/proc/20191002142203_13724_processed/denoise_xz_p2_ccpi_denoising_gpu.h5', 'r')
TomoRec3D = h5f['/2-CcpiDenoisingGpu-denoise_xz/data'][:]
h5f.close()

TomoRec3D = TomoRec3D/np.max(TomoRec3D)

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.imshow(TomoRec3D[440,:,:], vmin=0.0, vmax=0.25, cmap="gray")
plt.title('Denoised volume')
plt.show()

image = TomoRec3D[390:590,:,:]
slices, NxSize, NySize = np.shape(TomoRec3D)
#del TomoRec3D_13551
#%%1
"""
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
"""
#%%
print ("Geodesic distance calculation")
#initialise level sets
#ls1 = initialiseLS(slices, NxSize, NySize, coordX0=260, coordY0=243, coordZ0=0,\
#                   coordX1=362, coordY1=373, coordZ1=slices, circle_size = 10)

ls1 = initialiseLS(slices, NxSize, NySize, coordX0=394, coordY0=295, coordZ0=0,\
                   coordX1=394, coordY1=295, coordZ1=slices, circle_size = 10)


segm3D_geo = np.float32(np.zeros(np.shape(image)))
for j in range(0,slices):
    #segm3D_geo[j,:,:] = geodesic_distance.geodesic2d_fast_marching(NLTV_im[j,:,:],ls1[j,:,:])
    #using raster scan (faster)
    segm3D_geo[j,:,:] = geodesic_distance.geodesic2d_raster_scan(image[j,:,:], ls1[j,:,:], 0.5, 4)


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
plt.imshow(segm3D_geo[0,:,:], vmin=0.0, vmax=0.25, cmap="gray")
plt.pause(.2)

#%%
import numpy as np
from i23.methods.segmentation import MASK_ITERATE
import matplotlib.pyplot as plt

segm3D_geo = np.load('/home/kjy41806/Documents/tempGeo.npy')
MASK = np.load('/home/kjy41806/Documents/tempMask.npy')
image = np.load('/home/kjy41806/Documents/tempImage.npy')

MASK_upd = MASK_ITERATE(segm3D_geo[0,:,:], MASK[0,:,:], threhsold = 0.01, iterationsNumb = 50)

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.imshow(image[0,:,:], vmin=0.0, vmax=0.25, cmap="gray")
#%%
from sklearn.mixture import GaussianMixture
inputdata = segm3D_geo.reshape(slices*NxSize*NySize, 1)
total_classesNum = 3
#hist, bin_edges = np.histogram(inputdata, bins=100)
classif = GaussianMixture(n_components=total_classesNum, covariance_type="full")
classif.fit(inputdata)
cluster = classif.predict(inputdata)
segm = classif.means_[cluster]
segm = segm.reshape(slices,NxSize,NySize)
mask = segm.astype(np.float64) / np.max(segm)
mask = 255 * mask # Now scale by 255
mask = mask.astype(np.uint8) # we obtain the mask 

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('GMM segmented (clustered) image')
plt.imshow(mask[0,:,:])

#%%
from i23.methods.segmentation import MASK_CORR
# Now we process the mask 
mask2D = mask[0,:,:]
mask_input = mask2D.copy()
total_classesNum = 5

pars = {'maskdata' : mask_input,\
        'class_names': ('liquor','air','loop'),\
        'total_classesNum': total_classesNum,\
        'restricted_combinations': (('loop','crystal','liquor','loop'),
                                    ('air','artifacts','liquor','liquor'),
                                    ('air','loop','liquor','liquor'),
                                    ('air','artifacts','loop','loop'),
                                    ('air','crystal','loop','loop'),
                                    ('air','loop','crystal','crystal')),\
        'CorrectionWindow' : 10,\
        'iterationsNumb' : 6}

upd_mask_input = MASK_CORR(pars['maskdata'], pars['class_names'], \
                pars['total_classesNum'], pars['restricted_combinations'],\
                pars['CorrectionWindow'], pars['iterationsNumb'])

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Segmented image (postproc GMM)')
plt.imshow(upd_mask_input)
#%%

#%%