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

vert_tuple = [i for i in range(700,900)] # selection of vertical slice

h5py_list = h5py.File('/media/HD-LXU3/ITT_BATH_DLS/DataP_II_I23_alignment/rawdata/13068/13068.nxs','r')

darks = h5py_list['/entry1/instrument/flyScanDetector/data'][0:19,vert_tuple,:]
flats = h5py_list['/entry1/instrument/flyScanDetector/data'][20:39,vert_tuple,:]
#(proj_t, det_z, det_y) = np.shape(flats)

flats = np.swapaxes(np.swapaxes(flats,2,0),2,1)
darks = np.swapaxes(np.swapaxes(darks,2,0),2,1)

data_raw = h5py_list['/entry1/instrument/flyScanDetector/data'][40:None,vert_tuple,:]
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
# normalising the data
starind = 2
vert_select = [i for i in range(starind,starind+2)] # selection for normalaiser
data_norm = normaliser(data_raw[:,:,vert_select], flats[:,:,vert_select], darks[:,:,vert_select], log='log')

plt.figure()
plt.imshow(np.transpose(data_norm[:,:,0]), vmin=0, vmax=1.8, cmap="gray")
plt.title('Normalised projection')

# Reconstructing normalised data
N_size = 900
detectHoriz, anglesNum, slices = np.shape(data_norm)
det_y_crop = [i for i in range(77,detectHoriz-60)]

RectoolsDIR = RecToolsDIR(DetectorsDimH = np.size(det_y_crop),  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device='gpu')
FBP = RectoolsDIR.FBP(np.transpose(data_norm[det_y_crop,:,0]))

plt.figure()
plt.imshow(FBP, vmin=-0.001, vmax=0.005, cmap="gray")
plt.title('FBP reconstruction')
plt.show()
#%%
"""
h5f = h5py.File('i23_13068.h5', 'w')
h5f.create_dataset('data_norm', data=data_norm)
h5f.create_dataset('data_raw', data=data_raw)
h5f.create_dataset('flats', data=flats)
h5f.create_dataset('darks', data=darks)
h5f.create_dataset('angles_rad', data=angles_rad)
h5f.close()
"""
#%%
data_raw_norm = np.float32(np.transpose(data_raw[det_y_crop,:,vert_select[0]]))/np.max(np.float32(np.transpose(data_raw[det_y_crop,:,vert_select[0]])))

from tomobar.methodsIR import RecToolsIR
# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = np.size(det_y_crop),  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='PWLS',# data fidelity, choose LS, PWLS, GH (wip), Student (wip)
                    nonnegativity='ENABLE', # enable nonnegativity constraint (set to 'ENABLE')
                    OS_number = 12, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-09, # tolerance to stop outer iterations earlier
                    device='gpu')
lc = Rectools.powermethod(data_raw_norm) # calculate Lipschitz constant (run once to initilise)
#%%
RecFISTA = Rectools.FISTA(np.transpose(data_norm[det_y_crop,:,0]), \
                              weights=data_raw_norm, \
                              iterationsFISTA = 25, \
                              #huber_data_threshold = 0.04,\
                              student_data_threshold = 0.95,\
                              regularisation = 'FGP_TV', \
                              regularisation_parameter = 0.0000025,\
                              regularisation_iterations = 600,\
                              lipschitz_const = lc)

plt.figure()
plt.imshow(RecFISTA, vmin=-0.001, vmax=0.005, cmap="gray")
plt.title('Iterative FISTA-TV reconstruction')
plt.show()
#%%

