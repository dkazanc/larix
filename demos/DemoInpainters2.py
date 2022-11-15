#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created August 2022

Testing the capability of some morphological inpainting methods

@author: Daniil Kazantsev
"""

import matplotlib.pyplot as plt
import numpy as np
import timeit
from larix.methods.misc import INPAINT_NDF, INPAINT_EUCL_WEIGHTED
from larix.methods.misc import STRIPES_DETECT, STRIPES_MERGE

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
# 3d projection data
proj3d_data =  np.load('../data/sino_stripes_crop3D.npy')

sliceno = 6
plt.figure()
plt.imshow(proj3d_data[:,sliceno,:], cmap="gray", vmin= 0.8, vmax=1.3)
plt.title('Cropped sinogram')
plt.show()
#%%
# accenuate any stripes present in the data first (get weights)
(stripe_weights, stats_vec) = STRIPES_DETECT(proj3d_data, search_window_dims=(1,7,1), vert_window_size=5, gradient_gap=3)

# threshold weights to get a initialisation of the mask
threshold = 0.5 #larger more sensitive to stripes
mask_stripe = np.zeros_like(stripe_weights,dtype="uint8")
mask_stripe = np.ascontiguousarray(mask_stripe, dtype=np.uint8)
mask_stripe[stripe_weights > stats_vec[3]/threshold] = 1

# merge edges that are close to each other
mask_stripe_merged = STRIPES_MERGE(np.ascontiguousarray(mask_stripe, dtype=np.uint8), stripe_width_max_perc=25, dilate=3)

plt.figure()
plt.subplot(131)
plt.imshow(stripe_weights[:,sliceno,:])
plt.title('Stripe weights based on Gradient - X')
plt.subplot(132)
plt.imshow(mask_stripe[:,sliceno,:])
plt.title('Thresholded weights')
plt.subplot(133)
plt.imshow(mask_stripe_merged[:,sliceno,:])
plt.title('Processed mask')
plt.show()
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("___Inpainting in 2D using boundaries exatrapolation___")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
## plot 
fig = plt.figure()
plt.suptitle('Performance of ')
a=fig.add_subplot(1,2,1)
a.set_title('Missing data sinogram')
imgplot = plt.imshow(proj3d_data[:,sliceno,:],cmap="gray", vmin= 0.8, vmax=1.3)

# set parameters
pars = {'algorithm' : INPAINT_EUCL_WEIGHTED, 
        'input' : proj3d_data[:,sliceno,:],
        'maskData' : mask_stripe_merged[:,sliceno,:],
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
fig = plt.figure()
plt.suptitle('Performance of ')
a=fig.add_subplot(1,2,1)
a.set_title('Missing data sinogram')
imgplot = plt.imshow(proj3d_data[:,sliceno,:],cmap="gray", vmin= 0.8, vmax=1.3)

# set parameters
pars = {'algorithm' : INPAINT_EUCL_WEIGHTED, 
        'input' : proj3d_data,
        'maskData' : mask_stripe_merged,
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
plt.title('{}'.format('Extrapolation inpainting results'))%

#%%