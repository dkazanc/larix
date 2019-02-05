#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
# Reading I23 fly-scan () data with a drifting sample
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

# loading data 
h5f = h5py.File('../data/i23_12923.h5','r')
data_raw = h5f['data_raw'][:]
flats = h5f['flats'][:]
darks = h5f['darks'][:]
angles_rad = h5f['angles_rad'][:]
h5f.close()

#%%
"""
for i in range(0, 1801):
    data_diff[:,i,:] = data_raw[:,i,:] - flats[:,18,:]
"""
#%%
# normalising the data
from tomorec.supp.suppTools import normaliser
from tomorec.methodsDIR import RecToolsDIR

vert_select = [i for i in range(300,310)] # selection for normalaiser
data_norm = normaliser(data_raw[:,:,vert_select], flats[:,:,vert_select], darks[:,:,vert_select], log='log')

plt.figure()
plt.imshow(np.transpose(data_norm[:,:,0]), vmin=-0.3, vmax=0.9, cmap="gray")
plt.title('Normalised projection')
#%%
# Reconstructing normalised data
N_size = 1000
slice_no = 9
detectHoriz, anglesNum, slices = np.shape(data_norm)
det_y_crop = [i for i in range(200,detectHoriz-258)]

RectoolsDIR = RecToolsDIR(DetectorsDimH = np.size(det_y_crop),  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device='gpu')
FBP = RectoolsDIR.FBP(np.transpose(data_norm[det_y_crop,:,slice_no]))

plt.figure()
plt.imshow(FBP, vmin=0, vmax=0.003, cmap="gray")
plt.title('FBP reconstruction')
plt.show()
#%%
from tomorec.methodsIR import RecToolsIR

slice_no = 9
# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = np.size(det_y_crop),  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='PWLS',# data fidelity, choose LS, PWLS, GH (wip), Student (wip)
                    nonnegativity='on', # enable nonnegativity constraint (set to 'on')
                    OS_number = 12, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-09, # tolerance to stop outer iterations earlier
                    device='gpu')

rawdata_temp = np.float32(np.transpose(data_raw[det_y_crop,:,slice_no]))/np.float32(np.max(data_raw[det_y_crop,:,slice_no]))
lc = Rectools.powermethod(rawdata_temp) # calculate Lipschitz constant (run once to initialise)
#%%
RecFISTA_os_pwls = Rectools.FISTA(np.transpose(data_norm[det_y_crop,:,slice_no]), \
                             rawdata_temp,\
                             regularisation = 'FGP_TV', \
                             regularisation_parameter = 0.000002,\
                             regularisation_iterations = 150,\
                             iterationsFISTA = 15, \
                             lipschitz_const = lc)
plt.figure()
plt.imshow(RecFISTA_os_pwls, vmin=0, vmax=0.003, cmap="gray")
plt.title('FISTA-TV corr recon') 
plt.show()
#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt

# loading data 
h5f = h5py.File('../data/i23_12923.h5','r')
data_raw = h5f['data_raw'][:]
flats = h5f['flats'][:]
darks = h5f['darks'][:]
angles_rad = h5f['angles_rad'][:]
h5f.close()

#%%
from Ctypes_wrap import flatsregC

__flatreg__ = flatsregC() # initiate a function

#flats2D = flats[:,10,:]
flats3D = np.swapaxes(flats,0,1)
dark2D = np.uint16(np.average(darks,1)) # average of darks
drift_window = 10

flats_num = 19
#output = np.zeros(np.shape(flats2D))
drift_vector = np.float32(np.zeros((flats_num,(drift_window*2 + 1),(drift_window*2 + 1))))
norm_projection = np.float32(np.zeros(np.shape(dark2D)))
norm_projection_corr = np.float32(np.zeros(np.shape(dark2D)))
(detH, detV) = np.shape(dark2D)
flats3D = np.ascontiguousarray(flats3D, dtype=np.uint16);
dark2D = np.ascontiguousarray(dark2D, dtype=np.uint16);
drift_vector = np.ascontiguousarray(drift_vector, dtype=np.float32);
norm_projection = np.ascontiguousarray(norm_projection, dtype=np.float32);
data_norm_corr = np.float32(np.zeros(np.shape(data_raw)))

for i in range(0,1800):
    proj2D = data_raw[:,i,:]
    proj2D = np.ascontiguousarray(proj2D, dtype=np.uint16);
    norm_projection_corr = np.ascontiguousarray(norm_projection_corr, dtype=np.float32);
    
    __flatreg__(flats3D, dark2D, proj2D, 80, 50, 600, 220, drift_window, detV, detH, flats_num, drift_vector, norm_projection, norm_projection_corr)
    #__flatreg__(flats3D, dark2D, proj2D, 80, 930, 600, 1100, drift_window, detV, detH, flats_num, drift_vector, norm_projection, norm_projection_corr)
    data_norm_corr[:,i,:] = norm_projection_corr
    """
    plt.figure()
    plt.imshow(drift_vector[0,:,:], vmin=0, vmax=300, cmap="gray")
    plt.show()
    
    plt.figure()
    plt.imshow(norm_projection, vmin=-0.2, vmax=1.7, cmap="gray")
    plt.title('Not corrected projection') 
    plt.show()
    
    plt.figure()
    plt.imshow(norm_projection_corr, vmin=-0.2, vmax=1.7, cmap="gray")
    plt.title('corrected projection') 
    plt.show()
    """
#%%


#%%

