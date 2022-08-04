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
from larix.methods.misc import INPAINT_NDF, INPAINT_EUCL_WEIGHTED
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

# read sinogram and the mask
sinogram =  np.load('../data/sino_stripe_i12.npy')
mask = np.uint8(np.zeros(np.shape(sinogram)))
mask[:,185:215] = 1

sinogram[mask ==1] = 0.0

plt.figure(1)
plt.subplot(121)
plt.imshow(sinogram,vmin=0.0, vmax=1)
plt.title('Missing Data sinogram')
plt.subplot(122)
plt.imshow(mask)
plt.title('Mask')
plt.show()
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("___Inpainting using boundaries exatrapolation___")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
## plot 
fig = plt.figure()
plt.suptitle('Performance of ')
a=fig.add_subplot(1,2,1)
a.set_title('Missing data sinogram')
imgplot = plt.imshow(sinogram,cmap="gray")

# set parameters
pars = {'algorithm' : INPAINT_EUCL_WEIGHTED, 
        'input' : sinogram,
        'maskData' : mask,
        'number_of_iterations' : 15,
        'windowsize_half' : 5,
        'method_type' : 'random'}
        
start_time = timeit.default_timer()
(inp_simple, mask_upd) = INPAINT_EUCL_WEIGHTED(pars['input'],
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
imgplot = plt.imshow(inp_simple, cmap="gray")
plt.title('{}'.format('Extrapolation inpainting results'))
pars['number_of_iterations']
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