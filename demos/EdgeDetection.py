import h5py
import numpy as np
import matplotlib.pyplot as plt


h5f = h5py.File('TomoRec3D_13551.h5', 'r')
TomoRec3D_13551 = h5f['data'][:]
h5f.close()

#%%
from i23.methods.segmentation import EDGES_CRYSTAL
image = TomoRec3D_13551[100,10:None,:]
dx,dy = np.gradient(image)
grad_image = np.sqrt(dx**2 + dy**2)
outputEdges,rotateim = EDGES_CRYSTAL(grad_image, 30, 0.007, 50)

#%%