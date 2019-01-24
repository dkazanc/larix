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

vert_select = [i for i in range(150,160)] # selection for normalaiser
data_norm = normaliser(data_raw[:,:,vert_select], flats[:,:,vert_select], darks[:,:,vert_select], log='log')

plt.figure()
plt.imshow(np.transpose(data_norm[:,:,0]), vmin=0, vmax=0.8, cmap="gray")
plt.title('Normalised projection')
#%%
# Reconstructing normalised data
N_size = 1000
detectHoriz, anglesNum, slices = np.shape(data_norm)
det_y_crop = [i for i in range(0,detectHoriz-58)]

RectoolsDIR = RecToolsDIR(DetectorsDimH = np.size(det_y_crop),  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device='gpu')
FBP = RectoolsDIR.FBP(np.transpose(data_norm[det_y_crop,:,0]))

plt.figure()
plt.imshow(FBP, vmin=0, vmax=0.003, cmap="gray")
plt.title('FBP reconstruction')
plt.show()
#%%
from Ctypes_wrap import flatsregC

__flatreg__ = flatsregC() # initiate a class
__flatreg__()

#%%


