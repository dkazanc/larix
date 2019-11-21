#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
# reading and reconstructing I23 data, sample 13282, fly-scan (first experiments)
"""
# reading i23 data
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tomobar.supp.suppTools import normaliser
from tomobar.methodsDIR import RecToolsDIR

vert_tuple = [i for i in range(600,700)] # selection of vertical slice

h5py_list = h5py.File('/dls/i23/data/2019/nr23571-5/processing/tomography/rotated/13282/13282.nxs','r')

darks = h5py_list['/entry1/instrument/flyScanDetector/data'][0:40,vert_tuple,:]
flats = h5py_list['/entry1/instrument/flyScanDetector/data'][40:80,vert_tuple,:]

data_raw = h5py_list['/entry1/instrument/flyScanDetector/data'][80:1881,vert_tuple,:]
angles = h5py_list['/entry1/tomo_entry/data/rotation_angle'][:] # extract angles
angles_rad = angles[80:1881]*np.pi/180.0

h5py_list.close()
#%%
fig= plt.figure()
plt.rcParams.update({'font.size': 21})
plt.subplot(121)
plt.imshow(flats[39,:,:], vmin=250, vmax=47000, cmap="gray")
plt.title('Flat field image')
plt.subplot(122)
plt.imshow(darks[39,:,:], vmin=0, vmax=1000, cmap="gray")
plt.title('Dark field image')
plt.show()
#%%
# normalising the data
starind = 0
vert_select = [i for i in range(starind,starind+30)] # selection for normalaiser
data_norm = normaliser(data_raw[:,vert_select,:], flats[:,vert_select,:], darks[:,vert_select,:], log='log')

data_norm = np.swapaxes(np.swapaxes(data_norm,2,0),2,1)
plt.figure()
plt.imshow(data_norm[:,:,0], vmin=0, vmax=3.5, cmap="gray")
plt.title('Normalised projection')
#%%
# Reconstructing normalised data
N_size = 1000
slice_to_rec =10
detectHoriz, anglesNum, slices = np.shape(data_norm)
det_y_crop = [i for i in range(17,detectHoriz-70)]

RectoolsDIR = RecToolsDIR(DetectorsDimH = np.size(det_y_crop),  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = None,
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device='gpu')

FBP = RectoolsDIR.FBP(np.transpose(data_norm[det_y_crop,:,slice_to_rec]))

plt.figure()
plt.imshow(FBP, vmin=-0.001, vmax=0.005, cmap="gray")
plt.title('FBP reconstruction')
plt.show()
#%%
from tomobar.methodsIR import RecToolsIR
# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = np.size(det_y_crop),  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = None,
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='LS',# data fidelity, choose LS, PWLS, GH (wip), Student (wip)
                    nonnegativity='ENABLE', # enable nonnegativity constraint (set to 'ENABLE')
                    OS_number = 6, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-11, # tolerance to stop outer iterations earlier
                    device='gpu')
lc = Rectools.powermethod() # calculate Lipschitz constant (run once to initilise)
#%%
RecFISTA = Rectools.FISTA(np.transpose(data_norm[det_y_crop,:,slice_to_rec]), \
                              iterationsFISTA = 25, \
                              #huber_data_threshold = 0.05,\
                              #student_data_threshold = 0.95,\
                              regularisation = 'FGP_TV', \
                              regularisation_parameter = 0.000001,\
                              regularisation_iterations = 300,\
                              lipschitz_const = lc)

plt.figure()
plt.imshow(RecFISTA, vmin=-0.001, vmax=0.005, cmap="gray")
plt.title('Iterative FISTA-TV reconstruction')
plt.show()
#%%
RecFISTA_Huber = Rectools.FISTA(np.transpose(data_norm[det_y_crop,:,slice_to_rec]), \
                              iterationsFISTA = 25, \
                              huber_data_threshold = 0.03,\
                              regularisation = 'FGP_TV', \
                              regularisation_parameter = 0.000001,\
                              regularisation_iterations = 300,\
                              lipschitz_const = lc)

plt.figure()
plt.imshow(RecFISTA_Huber, vmin=-0.001, vmax=0.005, cmap="gray")
plt.title('Iterative FISTA-TV-Huber reconstruction')
plt.show()
#%%
RecFISTA_HuberRing = Rectools.FISTA(np.transpose(data_norm[det_y_crop,:,slice_to_rec]), \
                              iterationsFISTA = 25, \
                              huber_data_threshold = 0.03,\
                              ring_model_horiz_size = 9,\
                              regularisation = 'FGP_TV', \
                              regularisation_parameter = 0.000001,\
                              regularisation_iterations = 300,\
                              lipschitz_const = lc)

plt.figure()
plt.imshow(RecFISTA_HuberRing, vmin=-0.001, vmax=0.005, cmap="gray")
plt.title('Iterative FISTA-TV-Huber-Ring reconstruction')
plt.show()
#%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# swapping axis to satisfy the reconstructor
data_norm = np.swapaxes(data_norm,0,2)
#%%

# 3D reconstruction
from tomobar.methodsIR import RecToolsIR
# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = np.size(det_y_crop),  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = 30,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = 0.0,
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='LS',# data fidelity, choose LS, PWLS, GH (wip), Student (wip)
                    nonnegativity='ENABLE', # enable nonnegativity constraint (set to 'ENABLE')
                    OS_number = 6, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-11, # tolerance to stop outer iterations earlier
                    device='gpu')
lc = Rectools.powermethod() # calculate Lipschitz constant (run once to initilise)
#%%
RecFISTA_Huber3D = Rectools.FISTA(data_norm[:,:,det_y_crop], \
                              iterationsFISTA = 20, \
                              huber_data_threshold = 0.03,\
                              regularisation = 'ROF_TV', \
                              regularisation_parameter = 0.000007,\
                              regularisation_iterations = 300,\
                              lipschitz_const = lc)

plt.figure()
plt.imshow(RecFISTA_Huber3D[15,:,:], vmin=-0.001, vmax=0.005, cmap="gray")
plt.title('Iterative Huber-TV 3D recon')
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RecFISTA_HuberRing3D = Rectools.FISTA(data_norm[:,:,det_y_crop], \
                              iterationsFISTA = 20, \
                              huber_data_threshold = 0.03,\
                              ring_model_horiz_size= 9, \
                              ring_model_slices_size = 2,\
                              regularisation = 'ROF_TV', \
                              regularisation_parameter = 0.000007,\
                              regularisation_iterations = 300,\
                              lipschitz_const = lc)

plt.figure()
plt.imshow(RecFISTA_HuberRing3D[15,:,:], vmin=-0.001, vmax=0.005, cmap="gray")
plt.title('Iterative HuberRing-TV 3D recon')
plt.show()
#%%