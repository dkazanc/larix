import h5py
import numpy as np
import matplotlib.pyplot as plt


h5f = h5py.File('TomoRec3D_13551.h5', 'r')
TomoRec3D_13551 = h5f['data'][:]
h5f.close()

#image = TomoRec3D_13551[100,10:None,:]
#image = TomoRec3D_13551[130,160:460,200:430] # 130-slice
image = TomoRec3D_13551[130,:,:] # 130-slice
image_t = image.copy()
image_t = image_t/np.max(image_t)

#%%
# establish seeding point for GeoDist


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
import geodesic_distance
mask = np.uint8(np.zeros(np.shape(image_t)))
#mask[325,350] = 1
mask[300,180] = 1
D1 = geodesic_distance.geodesic2d_fast_marching(image_t,mask)
#%%
crystal = np.uint8(np.zeros(np.shape(image_t)))
crystal[D1 <= 0.1] = 1

plt.figure()
plt.imshow(crystal, vmin=0.0, vmax=1.0)
plt.show()
#%%
#D2 = geodesic_distance.geodesic2d_raster_scan(image_t,mask, 0.5, 2)
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
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk


#segm = acwe_ls2[5,:,:]
segm1 = segm.copy()
selem = disk(5)

#eroded = erosion(segm_close, selem)
#segm_dil = dilation(segm_close, selem)
segm_close = closing(segm1, selem)
plt.figure()
plt.imshow(segm_close, vmin=0, vmax=1.0, cmap="gray")
plt.show()

selem = disk(25)
eroded = erosion(segm_close, selem)
plt.figure()
plt.imshow(eroded, vmin=0, vmax=1.0, cmap="gray")
plt.show()

eroded[eroded == 0] = -1
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