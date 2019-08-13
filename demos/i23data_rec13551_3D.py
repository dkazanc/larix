#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
# reading and reconstructing I23 data, sample 13551, fly-scan 
"""
# reading i23 data
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tomobar.supp.suppTools import normaliser
from tomobar.methodsDIR import RecToolsDIR
from tomobar.methodsIR import RecToolsIR

vert_tuple = [i for i in range(650,1200)] # selection of vertical slice
#vert_tuple = [i for i in range(0,1600)] # selection of vertical slice

h5py_list = h5py.File('/dls/i23/data/2019/nr23571-10/processing/tomography/rotated/13551/13551.nxs','r')

darks = h5py_list['/entry1/instrument/flyScanDetector/data'][0:19,vert_tuple,:]
flats = h5py_list['/entry1/instrument/flyScanDetector/data'][20:39,vert_tuple,:]

data_raw = h5py_list['/entry1/instrument/flyScanDetector/data'][40:-1-20,vert_tuple,:]
angles = h5py_list['/entry1/tomo_entry/data/rotation_angle'][:] # extract angles
angles_rad = angles[40:-1-20]*np.pi/180.0

h5py_list.close()

fig= plt.figure()
plt.rcParams.update({'font.size': 21})
plt.subplot(131)
plt.imshow(flats[18,:,:], vmin=250, vmax=50000, cmap="gray")
plt.title('Flat field image')
plt.subplot(132)
plt.imshow(darks[18,:,:], vmin=0, vmax=1000, cmap="gray")
plt.title('Dark field image')
plt.subplot(133)
plt.imshow(data_raw[0,:,:],  cmap="gray")
plt.title('First raw projection')
plt.show()
#%%
# normalising the data
starind = 150 # index to start
addSlices = 20 # slices to add to start index
vert_select = [i for i in range(starind,starind+addSlices)] # selection for normalaiser
data_norm = normaliser(data_raw[:,vert_select,:], flats[:,vert_select,:], darks[:,vert_select,:], log='log')
#data_norm = normaliser(data_raw, flats, darks, log='log')

plt.figure()
plt.imshow(data_norm[0,:,:], vmin=0, vmax=1.8, cmap="gray")
plt.title('Normalised projection')
#%%
# One can crop automatically the normalised data
from tomobar.supp.suppTools import autocropper
cropped_data = autocropper(data_norm, addbox=20, backgr_pix1=20)

plt.figure()
plt.imshow(cropped_data[0,:,:], vmin=0, vmax=1.8, cmap="gray")
plt.title('Cropped normalised projection')

# swapping axis to satisfy the reconstructor
cropped_data = np.swapaxes(cropped_data,0,1)
slices, anglesNum, detectHorizCrop = np.shape(cropped_data)
#%%
del data_raw, darks, flats
#%%
# Reconstructing normalised data with FBP
RectoolsDIR = RecToolsDIR(DetectorsDimH = detectHorizCrop,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = slices,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = 5.5, # Center of Rotation (CoR) scalar (for 3D case only)
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = detectHorizCrop, # a scalar to define reconstructed object dimensions
                    device='gpu')

FBP = RectoolsDIR.FBP(cropped_data)

plt.figure()
plt.imshow(FBP[10,:,:], vmin=-0.001, vmax=0.015, cmap="gray")
plt.title('FBP reconstruction')
plt.show()
#%%
#data_raw_norm = cropped_data/np.max(cropped_data)
# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = detectHorizCrop,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = slices,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = 5.5, # Center of Rotation (CoR) scalar (for 3D case only)
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = detectHorizCrop, # a scalar to define reconstructed object dimensions
                    datafidelity='LS',# data fidelity, choose LS, PWLS, GH (wip), Student (wip)
                    nonnegativity='ENABLE', # enable nonnegativity constraint (set to 'ENABLE')
                    OS_number = 12, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-09, # tolerance to stop outer iterations earlier
                    device='gpu')

lc = Rectools.powermethod() # calculate Lipschitz constant (run once to initilise)
#%%
RecFISTA2 = Rectools.FISTA(cropped_data, \
                          iterationsFISTA = 15, \
#                          student_data_threshold = 0.95,\
                          regularisation = 'FGP_TV', \
                          regularisation_parameter = 0.0000015,\
                          regularisation_iterations = 350,\
                          lipschitz_const = lc)

plt.figure()
plt.imshow(RecFISTA2[10,:,:], vmin=-0.001, vmax=0.015, cmap="gray")
plt.title('Iterative FISTA-TV reconstruction')
plt.show()
#%%
"""
h5f = h5py.File('i23_13076.h5', 'w')
h5f.create_dataset('data_norm', data=data_norm)
#h5f.create_dataset('flats', data=flats)
#h5f.create_dataset('darks', data=darks)
h5f.create_dataset('angles_rad', data=angles_rad)
h5f.close()
"""
#%%
# Reading reconstructed data from h5 file !
h5py_list = h5py.File('tomo_p2_tomobar_recon_3D.h5','r')
TomobarRec3D = h5py_list['/2-TomobarRecon3d-tomo/data'][:]
TomobarRec3D = TomobarRec3D[:,:,:,0]
TomobarRec3D = TomobarRec3D/np.max(TomobarRec3D)
# 
TomobarRec3D = TomobarRec3D[280:550,:,150:750]

plt.figure()
plt.imshow(TomobarRec3D[50,:,:], vmin=0.0, vmax=0.25, cmap="gray")
plt.title('Iterative FISTA-TV reconstruction')
plt.show()

# save data
h5f = h5py.File('TomoRec3D_13551.h5', 'w')
h5f.create_dataset('data', data=TomobarRec3D)
h5f.close()

#read data 
import h5py
import numpy as np
import matplotlib.pyplot as plt
h5f = h5py.File('TomoRec3D_13551.h5', 'r')
TomoRec3D_13551 = h5f['data'][:]
h5f.close()
#%%
# using CV2 with Hough detection of lines
import cv2

image = TomoRec3D_13551[100,:,:]
image /= np.max(image)
kernel_size = 5
blur_gray = cv2.GaussianBlur(image,(kernel_size, kernel_size),0)
blur_gray /= np.max(blur_gray)

gray = (blur_gray*255).astype(np.uint8)
edges = cv2.Canny(gray, 1, 15, apertureSize = 3)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 70, maxLineGap=100)
for line in lines:
   x1, y1, x2, y2 = line[0]
   cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 128), 1)
cv2.imshow("linesEdges", edges)
cv2.imshow("linesDetected", gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#%%
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

image = TomoRec3D_13551[100,0:600,:]
image = image/np.max(image)

s = np.linspace(0, 2*np.pi, 800)
x = 300 + 200*np.cos(s)
y = 270 + 260*np.sin(s)
init = np.array([x, y]).T

snake = active_contour(gaussian(image, 3),
                       init, alpha=0.015, beta=0.5, gamma=0.001)

#%%
snake2 = snake.copy()
#snake2[:,0] = snake2[:,0] - 33*np.cos(s)
#snake2[:,1] = snake2[:,1] - 33*np.sin(s)
snake2[:,0] = snake2[:,0] - 0*np.cos(s)
snake2[:,1] = snake2[:,1] - 0*np.sin(s)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(image, cmap=plt.cm.gray)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake2[:, 0], snake2[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, image.shape[1], image.shape[0], 0])
plt.show()
#%%
image2 = image.copy()

cordY = np.uint16(snake2[:, 0])
cordX = np.uint16(snake2[:, 1])

image2[cordX, cordY] = 1.0

import scipy.ndimage as ndimage    

# Create an empty image to store the masked array
r_mask = np.zeros_like(r, dtype='bool')

# Create a contour image by using the contour coordinates rounded to their nearest integer value
r_mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1

# Fill in the hole created by the contour boundary
r_mask = ndimage.binary_fill_holes(r_mask)

# Invert the mask since you want pixels outside of the region
r_mask = ~r_mask

#%%