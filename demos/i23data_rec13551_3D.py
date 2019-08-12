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


h5f = h5py.File('TomoRec3D_13551.h5', 'w')
h5f.create_dataset('data', data=TomobarRec3D)
h5f.close()
#%%

