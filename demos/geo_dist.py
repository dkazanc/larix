import h5py
import numpy as np
import matplotlib.pyplot as plt


h5f = h5py.File('/home/kjy41806/Documents/data_temp/TomoRec3D_13551.h5', 'r')
TomoRec3D_13551 = h5f['data'][:]
h5f.close()

#image = TomoRec3D_13551[100,10:None,:]
#image = TomoRec3D_13551[130,160:460,200:430] # 130-slice
# image = TomoRec3D_13551[130,:,:] # 130-slice
image = TomoRec3D_13551[40:90,:,:] # 130-slice
image_t = image.copy()
image_t = image_t/np.max(image_t)

slices, NxSize, NySize = np.shape(image_t)
#%%
from ccpi.filters.regularisers import PatchSelect, NLTV

NLTV_im = np.float32(np.zeros(np.shape(image_t)))
print ("Doing NLTV denoising of X-Y slices")
for j in range(0,slices):
    # NLM processing of image
    pars = {'algorithm' : PatchSelect, \
            'input' : image_t[j,:,:],\
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
            'input' : image_t[j,:,:],\
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
#%%

import geodesic_distance
import timeit
mask = np.uint8(np.zeros(np.shape(image_t)))
mask[0:slices,300,300] = 1
start_time = timeit.default_timer()
Geo3D = geodesic_distance.geodesic3d_fast_marching(image_t,mask)
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
#%%
import geodesic_distance

#from scipy.ndimage import gaussian_filter
#image_t_blur = gaussian_filter(image_t, sigma=2.5)

mask = np.uint8(np.zeros(np.shape(NLTV_im)))
mask[0:slices,300,300] = 1
segm3D_as2D = np.float32(np.zeros(np.shape(NLTV_im)))
for j in range(0,slices):
    segm3D_as2D[j,:,:] = geodesic_distance.geodesic2d_fast_marching(NLTV_im[j,:,:],mask[j,:,:])

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('geodesic distance results')
plt.imshow(segm3D_as2D[5,:,:])
#%%
from sklearn.mixture import GaussianMixture
#from sklearn.mixture import BayesianGaussianMixture

Geo3D = segm3D_as2D.copy()

inputdata = Geo3D.reshape(slices*NxSize*NySize, 1)/np.max(Geo3D)
classes_number = 3
classif = GaussianMixture(n_components=classes_number, covariance_type="tied")
#classif = BayesianGaussianMixture(n_components=classes_number, covariance_type="tied")
classif.fit(inputdata)
cluster = classif.predict(inputdata)
segm = classif.means_[cluster]
segm = segm.reshape(slices, NxSize, NySize)
mask = segm.astype(np.float64) / np.max(segm)
mask = 255 * mask # Now scale by 255
mask = mask.astype(np.uint8) # we obtain the mask 

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('GMM segmented (clustered) image')
plt.imshow(mask[5,:,:])
#%%
#%%
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

selem = disk(8)
mask_morph = np.uint8(np.zeros(np.shape(mask)))

for j in range(0,10):
    segm_to_proc = mask[j,:,:].copy()
    eroded = erosion(segm_to_proc, selem)
    closing_t = closing(eroded, selem)
    segm_dil = dilation(closing_t, selem)
    mask_morph[j,:,:] = segm_dil

mask_morph[mask_morph != np.min(mask_morph)] = 0
plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Morphological processing of liquer/loop')
plt.imshow(mask_morph[5,:,:])

#%%
CrystalReg3D = Geo3D*mask_morph

slices, NxSize, NySize = np.shape(CrystalReg3D)
inputdata = CrystalReg3D.reshape(slices*NxSize*NySize, 1)/np.max(CrystalReg3D)
classes_number = 2
#hist, bin_edges = np.histogram(inputdata, bins=100)
classif = GaussianMixture(n_components=classes_number, covariance_type="tied")
#classif = BayesianGaussianMixture(n_components=classes_number, covariance_type="tied")
classif.fit(inputdata)
cluster = classif.predict(inputdata)
segm = classif.means_[cluster]
segm = segm.reshape(slices, NxSize, NySize)
mask2 = segm.astype(np.float64) / np.max(segm)
mask2 = 255 * mask2 # Now scale by 255
mask2 = mask2.astype(np.uint8) # we obtain the mask 

mask_morph[mask_morph == 0] = 100
mask2 = mask2*mask_morph

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('GMM segmented (clustered) crystal')
plt.imshow(mask2[5,:,:])
#%%
crystal3D = np.uint8(np.zeros(np.shape(mask)))
crystal3D[mask2 == np.min(mask2)] = 1

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk

selem = disk(10)
crystal_morph = np.uint8(np.zeros(np.shape(crystal3D)))

for j in range(0,slices):
    segm_to_proc = crystal3D[j,:,:].copy()
    closing_t = closing(segm_to_proc, selem)
    eroded = erosion(closing_t, selem)
    crystal_morph[j,:,:] = eroded

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Morphological processing of crystal')
plt.imshow(crystal_morph[5,:,:])
#%%

# trying 3D 
mask = np.uint8(np.zeros(np.shape(TomoRec3D_13551)))
mask[130,325,350] = 1
D_vol = geodesic_distance.geodesic3d_fast_marching(TomoRec3D_13551,mask)
#%%
from i23.methods.segmentation import GRAD_CENTRAL
import numpy.linalg as lin

dt = 0.1 # Scheme step
eta = 0.01 # Small value to avoid division by zero.
epsilon = 0.8 # Epsilon used for the delta approximation
alpha_in = 1.5 #  fidelity terms
alpha_out = 2.0
iterationsNumb = 800

f = image_t.copy()
mask = np.float32(np.zeros(np.shape(f)))
mask[:] = -1.0
mask[5:595,5:595] = 1.0

Phi = mask.copy()

for j in range(0,iterationsNumb):

    #calculate central differences
    Phi_x,Phi_y = GRAD_CENTRAL(Phi)
    Phi_xx,Phi_xy = GRAD_CENTRAL(Phi_x)
    Phi_yx,Phi_yy = GRAD_CENTRAL(Phi_y)
    
    # %Compute the values c_in and c_out
    in_1 =  np.sum((Phi>0.0)*f,0)
    in_2 = eta+np.sum((Phi>0.0),0)
    A = np.matrix(in_1)
    B = np.matrix(in_2)
    B = np.float32(B.T)
    c_in = np.linalg.lstsq(B,A.T, rcond=None)[0]
    
    out_1 =  np.sum((Phi<0.0)*f,0)
    out_2 = eta+np.sum((Phi<0.0),0)
    A = np.matrix(out_1)
    B = np.matrix(out_2)
    B = np.float32(B.T)
    c_out = np.linalg.lstsq(B,A.T, rcond=None)[0]
    
    #%TV term = Num/Den
    divgrad_num = Phi_xx*Phi_y**2 - 2.0*Phi_x*Phi_y*Phi_xy + Phi_yy*Phi_x**2
    divgrad_denom = (Phi_x**2 + Phi_y**2)**(1.5) + eta
    divgrad = divgrad_num/divgrad_denom
    
    #Update Phi
    Phi =  Phi + dt*epsilon/(np.pi*(epsilon**2 + Phi**2))*(divgrad - alpha_in*(f-np.array(c_in[0]))**2 + alpha_out*(f-np.array(c_out[0]))**2)

#segm = np.float32(np.zeros(np.shape(f)))
#segm[Phi > 0] = 1

plt.figure()
plt.imshow(Phi, vmin=0, vmax=1.0, cmap="gray")
plt.show()
#%%
#%%
dt = 0.1 # Scheme step
eta = 0.01 # Small value to avoid division by zero.
epsilon = 3.0 # Epsilon used for the delta approximation
alpha_in = 5.0 #  fidelity terms
alpha_out = 1.0
iterationsNumb = 500

Phi = eroded.copy()

for j in range(0,iterationsNumb):

    #calculate central differences
    Phi_x,Phi_y = GRAD_CENTRAL(Phi)
    Phi_xx,Phi_xy = GRAD_CENTRAL(Phi_x)
    Phi_yx,Phi_yy = GRAD_CENTRAL(Phi_y)
    
    # %Compute the values c_in and c_out
    in_1 =  np.sum((Phi>0.0)*f,0)
    in_2 = eta+np.sum((Phi>0.0),0)
    A = np.matrix(in_1)
    B = np.matrix(in_2)
    B = np.float32(B.T)
    c_in = np.linalg.lstsq(B,A.T, rcond=None)[0]
    
    out_1 =  np.sum((Phi<0.0)*f,0)
    out_2 = eta+np.sum((Phi<0.0),0)
    A = np.matrix(out_1)
    B = np.matrix(out_2)
    B = np.float32(B.T)
    c_out = np.linalg.lstsq(B,A.T, rcond=None)[0]
    
    #%TV term = Num/Den
    divgrad_num = Phi_xx*Phi_y**2 - 2.0*Phi_x*Phi_y*Phi_xy + Phi_yy*Phi_x**2
    divgrad_denom = (Phi_x**2 + Phi_y**2)**(1.5) + eta
    divgrad = divgrad_num/divgrad_denom
    
    #Update Phi
    Phi =  Phi + dt*epsilon/(np.pi*(epsilon**2 + Phi**2))*(divgrad - alpha_in*(f-np.array(c_in[0]))**2 + alpha_out*(f-np.array(c_out[0]))**2)
    
plt.figure()
plt.imshow(Phi, vmin=0, vmax=1.0, cmap="gray")
plt.show()
"""
segm_loop = np.float32(np.zeros(np.shape(f)))
segm_loop[Phi > 0] = 2

segm_loop = segm_loop - eroded
"""

# establish seeding point for GeoDist

#%%
from skimage.data import camera
from skimage.filters import roberts, sobel, sobel_h, sobel_v, scharr, \
    scharr_h, scharr_v, prewitt, prewitt_v, prewitt_h
from skimage.filters import gaussian

edge_roberts = roberts(image_t)
edge_sobel = sobel(image_t)

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                       figsize=(8, 4))

ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
ax[0].set_title('Roberts Edge Detection')

ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
ax[1].set_title('Sobel Edge Detection')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()


dx,dy = np.gradient(image_t)
grad_image = np.sqrt(dx**2 + dy**2)

#%%
crystal = np.uint8(np.zeros(np.shape(image_t)))
crystal[D1 <= 0.1] = 1

plt.figure()
plt.imshow(crystal, vmin=0.0, vmax=1.0)
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#%
from sklearn.mixture import GaussianMixture
#from sklearn.mixture import BayesianGaussianMixture

######## first segmenting the object + vacuum (background) ########
classes_number = 2

wholeObj_SEG = np.uint8(np.zeros(np.shape(segm3D_as2D)))

for j in range(0,slices):
    inputdata = segm3D_as2D[j,:,:].reshape(NxSize*NySize, 1)/np.max(segm3D_as2D[j,:,:])
    classif = GaussianMixture(n_components=classes_number, covariance_type="full")
    classif.fit(inputdata)
    cluster = classif.predict(inputdata)
    segm = classif.means_[cluster]
    segm = segm.reshape(NxSize, NySize)
    mask = segm.astype(np.float64) / np.max(segm)
    mask = 255 * mask # Now scale by 255
    wholeObj_SEG[j,:,:] = mask.astype(np.uint8) # we obtain the mask 

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('GMM segmented (clustered) image')
plt.imshow(wholeObj_SEG[5,:,:])
#%%
# morphologically process it
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

selem = disk(10)
wholeObj_SEG_morph = np.uint8(np.zeros(np.shape(wholeObj_SEG)))

for j in range(0,slices):
    segm_to_proc = wholeObj_SEG[j,:,:].copy()
    closing_t = closing(segm_to_proc, selem)
    segm_tmp = opening(closing_t, selem)
    segm_tmp[segm_tmp != np.min(segm_tmp)] = 0
    segm_tmp[segm_tmp > 0] = 1
    wholeObj_SEG_morph[j,:,:] = segm_tmp

#%
plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Morphological processing of the whole object')
plt.imshow(wholeObj_SEG_morph[5,:,:])
#%%
# correct the data 
segm3D_as2D_obj = segm3D_as2D*wholeObj_SEG_morph
#%%
######## segmenting the loop  ########
classes_number = 2

wholeObj_SEG2 = np.uint8(np.zeros(np.shape(segm3D_as2D)))

for j in range(0,slices):
    inputdata = segm3D_as2D_obj[j,:,:].reshape(NxSize*NySize, 1)/np.max(segm3D_as2D_obj[j,:,:])
    classif = GaussianMixture(n_components=classes_number, covariance_type="tied")
    classif.fit(inputdata)
    cluster = classif.predict(inputdata)
    segm = classif.means_[cluster]
    segm = segm.reshape(NxSize, NySize)
    mask = segm.astype(np.float64) / np.max(segm)
    mask = 255 * mask # Now scale by 255
    wholeObj_SEG2[j,:,:] = mask.astype(np.uint8) # we obtain the mask 

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('GMM segmented (clustered) image')
plt.imshow(wholeObj_SEG2[5,:,:])

#%%

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

selem = disk(8)
mask_morph = np.uint8(np.zeros(np.shape(mask)))

for j in range(0,10):
    segm_to_proc = mask[j,:,:].copy()
    eroded = erosion(segm_to_proc, selem)
    closing_t = closing(eroded, selem)
    segm_dil = dilation(closing_t, selem)
    mask_morph[j,:,:] = segm_dil

mask_morph[mask_morph != np.min(mask_morph)] = 0
plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Morphological processing of liquer/loop')
plt.imshow(mask_morph[5,:,:])

#%%
CrystalReg3D = Geo3D*mask_morph

slices, NxSize, NySize = np.shape(CrystalReg3D)
inputdata = CrystalReg3D.reshape(slices*NxSize*NySize, 1)/np.max(CrystalReg3D)
classes_number = 2
#hist, bin_edges = np.histogram(inputdata, bins=100)
classif = GaussianMixture(n_components=classes_number, covariance_type="tied")
#classif = BayesianGaussianMixture(n_components=classes_number, covariance_type="tied")
classif.fit(inputdata)
cluster = classif.predict(inputdata)
segm = classif.means_[cluster]
segm = segm.reshape(slices, NxSize, NySize)
mask2 = segm.astype(np.float64) / np.max(segm)
mask2 = 255 * mask2 # Now scale by 255
mask2 = mask2.astype(np.uint8) # we obtain the mask 

mask_morph[mask_morph == 0] = 100
mask2 = mask2*mask_morph

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('GMM segmented (clustered) crystal')
plt.imshow(mask2[5,:,:])
#%%
crystal3D = np.uint8(np.zeros(np.shape(mask)))
crystal3D[mask2 == np.min(mask2)] = 1

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk

selem = disk(10)
crystal_morph = np.uint8(np.zeros(np.shape(crystal3D)))

for j in range(0,slices):
    segm_to_proc = crystal3D[j,:,:].copy()
    closing_t = closing(segm_to_proc, selem)
    eroded = erosion(closing_t, selem)
    crystal_morph[j,:,:] = eroded

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Morphological processing of crystal')
plt.imshow(crystal_morph[5,:,:])
#%%

# trying 3D 
mask = np.uint8(np.zeros(np.shape(TomoRec3D_13551)))
mask[130,325,350] = 1
D_vol = geodesic_distance.geodesic3d_fast_marching(TomoRec3D_13551,mask)
#%%
#distance calc
x0 = 50
y0 = 100
x1 = 80
y1 = 150

steps = 200
distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
d_dist = distance/(steps-1)
x_t = np.zeros(steps)
y_t = np.zeros(steps)

d_step = d_dist
for j in range(0,steps):
    t = d_step/distance
    x_t[j] = np.round((1.0 - t)*x0 + t*x1)
    y_t[j] = np.round((1.0 - t)*y0 + t*y1)
    d_step += d_dist
#%%