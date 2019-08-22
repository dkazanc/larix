import h5py
import numpy as np
import matplotlib.pyplot as plt


h5f = h5py.File('TomoRec3D_13551.h5', 'r')
TomoRec3D_13551 = h5f['data'][:]
h5f.close()

#image = TomoRec3D_13551[100,10:None,:]
image = TomoRec3D_13551[130,160:460,200:430] # 130-slice
#%%
image_t = image.copy()
from skimage.filters import frangi, hessian
frimage = frangi(image_t, beta1 = 1, beta2 = 0.01)

dx,dy = np.gradient(frimage)
grad_image = np.sqrt(dx**2 + dy**2)
theta = np.zeros(np.shape(grad_image))
theta[dx!=0] = np.arctan(dy[dx!=0]/dx[dx!=0]);

#%%
hist,bin_edges = np.histogram(theta, bins = 10)

#plt.figure(figsize=[10,8])

plt.figure()
plt.bar(bin_edges[:-1], hist, width = 0.1, color='#0504aa',alpha=0.7)
plt.xlim(min(bin_edges), max(bin_edges))
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.title('Normal Distribution Histogram',fontsize=15)
plt.show()


"""
plt.figure()
plt.hist(theta, bins=30);
plt.show()
"""
#%%
from i23.methods.segmentation import EDGES_CRYSTAL
outputEdges,rotateim = EDGES_CRYSTAL(grad_image, 30, 0.007, 50)

#%%