#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created August 2022

Testing the capability of some inpainting methods

@author: Daniil Kazantsev
"""

import matplotlib.pyplot as plt
import numpy as np
import timeit
#from larix.methods.misc import INPAINT_NDF, INPAINT_EUCL_WEIGHTED

###############################################################################
def printParametersToString(pars):
        txt = r''
        for key, value in pars.items():
            if key== 'algorithm' :
                txt += "{0} = {1}".format(key, value.__name__)
            elif key == 'input':
                txt += "{0} = {1}".format(key, np.shape(value))
            elif key == 'maskData':
                txt += "{0} = {1}".format(key, np.shape(value))
            else:
                txt += "{0} = {1}".format(key, value)
            txt += '\n'
        return txt
###############################################################################

def _gradient(data, axis):
    return np.gradient(data, axis=axis)


# 3d projection data
proj3d_data =  np.load('../data/sino_stripes_crop3D.npy')
mask3d_data = np.zeros_like(proj3d_data,dtype="uint8")

#detecting stripes to generate a mask
grad_data = _gradient(proj3d_data,axis=1)
sum_grad = np.zeros((np.size(grad_data,1),np.size(grad_data,2)))


from scipy.signal import find_peaks
from skimage.morphology import disk, binary_dilation
footprint = disk(7)

for i in range(np.size(grad_data,1)):
    sum_grad1d = np.sum(grad_data[:,i,:],0)
    sum_grad[i,:] = sum_grad1d/np.max(sum_grad1d)
    peaks = find_peaks(sum_grad[i,:], height = 0.1, distance = 1)
    get_peaks_indices = peaks[0]
    if len(get_peaks_indices) == 2:
        mask3d_data[:,i,get_peaks_indices[0]:get_peaks_indices[1]] = 1  
    else:
        mask3d_data[:,i,get_peaks_indices[0]] = 1  
    mask3d_data[:,i,:] = binary_dilation(mask3d_data[:,i,:], footprint)
    
sliceno = 5
plt.figure()
plt.subplot(121)
plt.imshow(proj3d_data[:,sliceno,:])
plt.title('Missing Data sinogram')
plt.subplot(122)
plt.imshow(mask3d_data[:,sliceno,:])
plt.title('Mask')
plt.show()

#%%

from larix.methods.misc import STRIPES_DETECT

stripe_weights = STRIPES_DETECT(np.ascontiguousarray(proj3d_data[:,5,:], dtype=np.float32), (1,7,1), "gradient")

plt.figure()
plt.imshow(stripe_weights)
plt.show()
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("___Inpainting in 2D using boundaries exatrapolation___")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
## plot 
sliceno = 3  
fig = plt.figure()
plt.suptitle('Performance of ')
a=fig.add_subplot(1,2,1)
a.set_title('Missing data sinogram')
imgplot = plt.imshow(proj3d_data[:,sliceno,:],cmap="gray", vmin= 0.8, vmax=1.3)

# set parameters
pars = {'algorithm' : INPAINT_EUCL_WEIGHTED, 
        'input' : np.ascontiguousarray(proj3d_data[:,sliceno,:], dtype=np.float32),
        'maskData' : np.ascontiguousarray(mask3d_data[:,sliceno,:], dtype=np.uint8),
        'number_of_iterations' : 3,
        'windowsize_half' : 4,
        'method_type' : 'random'}
        
start_time = timeit.default_timer()
inp_simple = INPAINT_EUCL_WEIGHTED(pars['input'],
              pars['maskData'], 
              pars['number_of_iterations'],
              pars['windowsize_half'],
              pars['method_type'])
              
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,2,2)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.1, 0.1, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(inp_simple, cmap="gray", vmin= 0.8, vmax=1.3)
plt.title('{}'.format('Extrapolation inpainting results'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("___Inpainting in 3D using boundaries exatrapolation___")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
## plot 
sliceno = 7  
fig = plt.figure()
plt.suptitle('Performance of ')
a=fig.add_subplot(1,2,1)
a.set_title('Missing data sinogram')
imgplot = plt.imshow(proj3d_data[:,sliceno,:],cmap="gray", vmin= 0.8, vmax=1.3)

# set parameters
pars = {'algorithm' : INPAINT_EUCL_WEIGHTED, 
        'input' : np.ascontiguousarray(proj3d_data, dtype=np.float32),
        'maskData' : np.ascontiguousarray(mask3d_data, dtype=np.uint8),
        'number_of_iterations' : 2,
        'windowsize_half' : 5,
        'method_type' : 'random'}
        
start_time = timeit.default_timer()
inp_simple = INPAINT_EUCL_WEIGHTED(pars['input'],
              pars['maskData'], 
              pars['number_of_iterations'],
              pars['windowsize_half'],
              pars['method_type'])
              
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,2,2)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.1, 0.1, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(inp_simple[:,sliceno,:], cmap="gray", vmin= 0.8, vmax=1.3)
plt.title('{}'.format('Extrapolation inpainting results'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("___Inpainting using linear diffusion (2D)__")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure(3)
plt.suptitle('Performance of linear inpainting using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Missing data sinogram')
imgplot = plt.imshow(sinogram,cmap="gray")

# set parameters
pars = {'algorithm' : INPAINT_NDF, \
        'input' : sinogram,\
        'maskData' : mask,\
        'regularisation_parameter':5000,\
        'edge_parameter':0,\
        'number_of_iterations' :7000 ,\
        'time_marching_parameter':0.000075,\
        'penalty_type':1
        }
        
start_time = timeit.default_timer()
ndf_inp_linear = INPAINT_NDF(pars['input'],
              pars['maskData'],
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'], 
              pars['penalty_type'])

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,2,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.1, 0.1, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(ndf_inp_linear, cmap="gray")
plt.title('{}'.format('Linear diffusion inpainting results'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_Inpainting using nonlinear diffusion (2D)_")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure(4)
plt.suptitle('Performance of nonlinear diffusion inpainting using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Missing data sinogram')
imgplot = plt.imshow(sinogram,cmap="gray")

# set parameters
pars = {'algorithm' : INPAINT_NDF, \
        'input' : sinogram,\
        'maskData' : mask,\
        'regularisation_parameter':80,\
        'edge_parameter':0.00009,\
        'number_of_iterations' :500 ,\
        'time_marching_parameter':0.000008,\
        'penalty_type':1
        }
        
start_time = timeit.default_timer()
ndf_inp_nonlinear = INPAINT_NDF(pars['input'],
              pars['maskData'],
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'], 
              pars['penalty_type'])
             

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,2,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.1, 0.1, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(ndf_inp_nonlinear, cmap="gray")
plt.title('{}'.format('Nonlinear diffusion inpainting results'))
#%%