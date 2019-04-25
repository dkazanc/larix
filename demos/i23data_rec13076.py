#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
# reading and reconstructing I23 data, sample 12923, fly-scan (first experiments)
"""

# reading i23 data

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tomobar.supp.suppTools import normaliser
from tomobar.methodsDIR import RecToolsDIR


vert_tuple = [i for i in range(550,650)] # selection of vertical slice

h5py_list = h5py.File('../rawdata/13076/13076.nxs','r')

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
starind = 50
vert_select = [i for i in range(starind,starind+2)] # selection for normalaiser
data_norm = normaliser(data_raw[:,:,vert_select], flats[:,:,vert_select], darks[:,:,vert_select], log='log')

plt.figure()
plt.imshow(np.transpose(data_norm[:,:,0]), vmin=0, vmax=1.8, cmap="gray")
plt.title('Normalised projection')

# Reconstructing normalised data
N_size = 700
detectHoriz, anglesNum, slices = np.shape(data_norm)
det_y_crop = [i for i in range(150,detectHoriz-201)]

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
h5f = h5py.File('i23_13076.h5', 'w')
h5f.create_dataset('data_norm', data=data_norm)
#h5f.create_dataset('flats', data=flats)
#h5f.create_dataset('darks', data=darks)
h5f.create_dataset('angles_rad', data=angles_rad)
h5f.close()
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
#lc = Rectools.powermethod() # calculate Lipschitz constant (run once to initilise)
lc = Rectools.powermethod(data_raw_norm) # calculate Lipschitz constant (run once to initilise)
#%%
RecFISTA_TV_os = Rectools.FISTA(np.transpose(data_norm[det_y_crop,:,0]), \
                              weights=data_raw_norm,\
                              iterationsFISTA = 15, \
                              regularisation = 'SB_TV', \
                              regularisation_parameter = 0.000003,\
                              regularisation_iterations = 500,\
                              lipschitz_const = lc)

plt.figure()
plt.imshow(RecFISTA_TV_os, vmin=0, vmax=0.008, cmap="gray")
plt.show()
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

[RecFISTA_Huber_TV_os,upd_mask] = Rectools.FISTA(np.transpose(data_norm[det_y_crop,:,0]), \
                              weights=data_raw_norm,\
                              iterationsFISTA = 20, \
                              #huber_data_threshold = 0.4,\
                              #student_data_threshold = 0.89,\
                              regularisation = 'FGP_TV', \
                              regularisation_parameter = 0.00001,\
                              regularisation_iterations = 600,\
                              time_marching_parameter = 0.01,\
                              lipschitz_const = lc)

plt.figure()
plt.imshow(RecFISTA_Huber_TV_os, vmin=-0.001, vmax=0.005, cmap="gray")
plt.title('Iterative FISTA-TV reconstruction')
plt.show()
#%%

#%%
from ccpi.filters.regularisers import PatchSelect

print ("Pre-calculating weights for non-local patches using FBP image...")

pars = {'algorithm' : PatchSelect, \
        'input' : FBP,\
        'searchwindow': 7, \
        'patchwindow': 2,\
        'neighbours' : 15 ,\
        'edge_parameter':0.0008}
H_i, H_j, Weights = PatchSelect(pars['input'], pars['searchwindow'],  pars['patchwindow'],         pars['neighbours'],
              pars['edge_parameter'],'gpu')

plt.figure()
plt.imshow(Weights[0,:,:], vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
#%%
from tomorec.methodsIR import RecToolsIR
# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = np.size(det_y_crop),  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='Huber',# data fidelity, choose LS, PWLS, GH (wip), Student (wip)
                    nonnegativity='ENABLE', # enable nonnegativity constraint (set to 'ENABLE')
                    OS_number = 12, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-09, # tolerance to stop outer iterations earlier
                    device='gpu')
lc = Rectools.powermethod() # calculate Lipschitz constant (run once to initilise)
#%%
RecFISTA_NLTV_os = Rectools.FISTA(np.transpose(data_norm[det_y_crop,:,0]), 
                              huber_data_threshold = 0.017,\
                              iterationsFISTA = 12, \
                              regularisation = 'NLTV', \
                              regularisation_parameter = 0.000005,\
                              regularisation_iterations = 30,\
                              NLTV_H_i = H_i,\
                              NLTV_H_j = H_j,\
                              NLTV_Weights = Weights,\
                              lipschitz_const = lc)
plt.figure()
plt.imshow(RecFISTA_NLTV_os, vmin=0, vmax=0.008, cmap="gray")
plt.show()
#%%


