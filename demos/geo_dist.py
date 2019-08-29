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

import geodesic_distance
mask = np.uint8(np.zeros(np.shape(image_t)))
mask[300,300] = 1
D1 = geodesic_distance.geodesic2d_fast_marching(image_t,mask)
#%%
crystal = np.uint8(np.zeros(np.shape(image_t)))
crystal[D1 <= 0.1] = 1
#%%
#D2 = geodesic_distance.geodesic2d_raster_scan(image_t,mask, 0.5, 2)
#%%
