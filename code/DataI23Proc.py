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

flats2D = flats[:,10,:]
proj2D = data_raw[:,0,:]
drift_window = 15

#output = np.zeros(np.shape(flats2D))
drift_vector = np.float32(np.zeros(((drift_window*2 + 1),(drift_window*2 + 1))))
(detH, detV) = np.shape(flats2D)

#output = np.ascontiguousarray(output, dtype=np.uint16);
flats2D = np.ascontiguousarray(flats2D, dtype=np.uint16);
proj2D = np.ascontiguousarray(proj2D, dtype=np.uint16);
drift_vector = np.ascontiguousarray(drift_vector, dtype=np.float32);

__flatreg__(flats2D, proj2D, 80, 50, 600, 200, drift_window, detV, detH, drift_vector)

plt.figure()
plt.imshow(drift_vector, vmin=0, vmax=5000, cmap="gray")
plt.show()
#%%


