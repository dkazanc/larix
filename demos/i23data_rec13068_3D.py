#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
# reading and reconstructing I23 data, sample 13068, fly-scan (first experiments)
"""
# reading i23 data
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tomobar.supp.suppTools import normaliser
from tomobar.methodsDIR import RecToolsDIR
from tomobar.methodsIR import RecToolsIR

vert_tuple = [i for i in range(700,900)] # selection of vertical slice

h5py_list = h5py.File('/dls/i23/data/2019/nr23017-1/processing/tomography/rotated/13068/13068.nxs','r')

darks = h5py_list['/entry1/instrument/flyScanDetector/data'][0:19,vert_tuple,:]
flats = h5py_list['/entry1/instrument/flyScanDetector/data'][20:39,vert_tuple,:]
flats = np.swapaxes(np.swapaxes(flats,2,0),2,1)
darks = np.swapaxes(np.swapaxes(darks,2,0),2,1)

data_raw = h5py_list['/entry1/instrument/flyScanDetector/data'][40:None,vert_tuple,:]
data_raw = np.swapaxes(np.swapaxes(data_raw,2,0),2,1)
angles = h5py_list['/entry1/tomo_entry/data/rotation_angle'][:] # extract angles
angles_rad = angles[40:None]*np.pi/180.0

h5py_list.close()
#%%
#vert_tuple = [i for i in range(700,900)] # selection of vertical slice

h5py_list = h5py.File('/dls/i23/data/2019/nr23017-1/processing/tomography/rotated/13068/13068.nxs','r')

darks = h5py_list['/entry1/instrument/flyScanDetector/data'][0:19,:,:]
flats = h5py_list['/entry1/instrument/flyScanDetector/data'][20:39,:,:]
flats = np.swapaxes(np.swapaxes(flats,2,0),2,1)
darks = np.swapaxes(np.swapaxes(darks,2,0),2,1)

data_raw = h5py_list['/entry1/instrument/flyScanDetector/data'][800:900,:,:]
data_raw = np.swapaxes(np.swapaxes(data_raw,2,0),2,1)
angles = h5py_list['/entry1/tomo_entry/data/rotation_angle'][:] # extract angles
angles_rad = angles[40:None]*np.pi/180.0

h5py_list.close()
#%%
fig= plt.figure()
plt.rcParams.update({'font.size': 21})
plt.subplot(121)
plt.imshow(flats[:,18,:], vmin=250, vmax=37000, cmap="gray")
plt.title('Flat field image')
plt.subplot(122)
plt.imshow(darks[:,18,:], vmin=0, vmax=1000, cmap="gray")
plt.title('Dark field image')
plt.show()
#%%
# cropping the 3D data: 
N_size = 770
detectHoriz, anglesNum, slices = np.shape(data_raw)
det_y_crop = [i for i in range(77,detectHoriz-60)]
data_raw = data_raw[det_y_crop,:,:]
flats = flats[det_y_crop,:,:]
darks = darks[det_y_crop,:,:]
#%%
# normalising the data
starind = 0
endind = 1599
vert_select = [i for i in range(starind,starind+endind)] # selection for normalaiser
data_norm = normaliser(data_raw[:,:,vert_select], flats[:,:,vert_select], darks[:,:,vert_select], log='log')

plt.figure()
plt.imshow(np.transpose(data_norm[:,:,0]), vmin=0, vmax=1.8, cmap="gray")
plt.title('Normalised projection')

detectHorizCrop, anglesNum, slices = np.shape(data_norm)
#%%
# Reconstructing normalised data with FBP
RectoolsDIR = RecToolsDIR(DetectorsDimH = detectHorizCrop,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = slices,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device='gpu')
FBP = RectoolsDIR.FBP(np.swapaxes(data_norm,2,0))

plt.figure()
plt.imshow(FBP[10,:,:], vmin=-0.001, vmax=0.005, cmap="gray")
plt.title('FBP reconstruction')
plt.show()
#%%
data_raw_norm = np.float32(np.transpose(data_raw[:,:,starind:starind+endind]))/np.max(np.float32(np.transpose(data_raw[:,:,starind:starind+endind])))

# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = detectHorizCrop,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = slices,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = None, # Center of Rotation (CoR) scalar (for 3D case only)
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='PWLS',# data fidelity, choose LS, PWLS, GH (wip), Student (wip)
                    nonnegativity='ENABLE', # enable nonnegativity constraint (set to 'ENABLE')
                    OS_number = 12, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-09, # tolerance to stop outer iterations earlier
                    device='gpu')

lc = Rectools.powermethod(data_raw_norm) # calculate Lipschitz constant (run once to initilise)
del data_raw, darks, flats
#%%
RecFISTA = Rectools.FISTA(np.swapaxes(data_norm, 2, 0), \
                              weights=data_raw_norm, \
                              iterationsFISTA = 20, \
                              #student_data_threshold = 0.95,\
                              regularisation = 'FGP_TV', \
                              regularisation_parameter = 0.0000015,\
                              regularisation_iterations = 450,\
                              lipschitz_const = lc)

plt.figure()
plt.imshow(RecFISTA[10,:,:], vmin=-0.001, vmax=0.005, cmap="gray")
plt.title('Iterative FISTA-TV reconstruction')
plt.show()
#%%
# np.save('13068_recon100slices.npy', RecFISTA)