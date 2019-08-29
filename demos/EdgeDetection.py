import h5py
import numpy as np
import matplotlib.pyplot as plt


h5f = h5py.File('TomoRec3D_13551.h5', 'r')
TomoRec3D_13551 = h5f['data'][:]
h5f.close()

#image = TomoRec3D_13551[100,10:None,:]
image = TomoRec3D_13551[130,160:460,200:430] # 130-slice


#%%
#image_t = image.copy()
#image_t = image_t/np.max(image_t)
image_t = nltv_cpu.copy()
#from skimage.filters import frangi, hessian
#frimage = frangi(image_t, beta1 = 1, beta2 = 0.01)

dx,dy = np.gradient(image_t[:,0:250])
grad_image = np.sqrt(dx**2 + dy**2)
theta = np.zeros(np.shape(grad_image))
theta[dx!=0] = np.arctan(dy[dx!=0]/dx[dx!=0]);
theta = theta + np.pi*0.5
#%%
bins_num = 20
hist,bin_edges = np.histogram(theta, bins = bins_num)
#plt.figure(figsize=[10,8])
plt.figure()
plt.bar(bin_edges[:-1], hist, width = np.abs(bin_edges[0] - bin_edges[1]), color='#0504aa',alpha=0.7)
plt.xlim(min(bin_edges), max(bin_edges))
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.title('Normal Distribution Histogram',fontsize=15)
plt.show()

#%%
#sorting histogram 
sort_hist = np.sort(hist)

image_segm = np.uint8(np.zeros(np.shape(theta)))

for i in range(0, 3):
    sel_hist = sort_hist[bins_num-1-i]
    index_sel = np.where(hist == sel_hist)
    min_range = bin_edges[index_sel[0]]
    max_range = bin_edges[index_sel[0]+1]
    image_segm[(theta >= min_range) & (theta < max_range)] = 1

#%%

#%%
"""
plt.figure()
plt.hist(theta, bins=30);
plt.show()
"""
#%%
image = TomoRec3D_13551[130,150:470,180:500] # 130-slice
#image = np.float32(np.zeros([300,300]))
#image[120:200,120:200] = 1.0
image_t = image.copy()

image_t = image_t/np.max(image_t)
from i23.methods.segmentation import EDGES_CRYSTAL
outputEdges,rotateim = EDGES_CRYSTAL(image_t, LineSize = 10, threshold = 0.007, OreintNo = 50)

plt.figure()
plt.imshow(rotateim, vmin=0.0, vmax=0.001, cmap="gray")
plt.show()
#%%
from skimage.filters import gabor
from skimage import data, io

image_t = image.copy()
angles_num = 100
angles = np.linspace(0.0,np.pi/2.0,angles_num,dtype='float32') 
filt_real3D = np.zeros([angles_num,300,230])
for i in range(0, angles_num):
    filt_real3D[i,:,:], filt_imag = gabor(image_t, frequency=0.1, theta=angles[i])


plt.figure()
io.imshow(filt_real3D[0,:,:])
io.show()


#%%
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import skeletonize, convex_hull_image
from skimage.morphology import disk

sumEdges_Edges2 = sumEdges_Edges.copy()
selem = disk(20)
sumEdges_Edges_res = skeletonize(sumEdges_Edges2)


#%%
#h5f = h5py.File('imageCryst.h5', 'w')
#h5f.create_dataset('data', data=image_t)
#h5f.close()



#%%