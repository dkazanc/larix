#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing the capability of stripe detection methods and also 
morphological inpainting methods

@author: Daniil Kazantsev
"""

import matplotlib.pyplot as plt
import numpy as np
import timeit
from larix.methods.misc import INPAINT_EUCL_WEIGHTED
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
data =  np.load('../data/proj3d_stripes.npz')
proj3d = data['proj3d']


sliceno = 5
plt.figure()
plt.imshow(proj3d[:,sliceno,:], cmap="gray", vmin= 0.5, vmax=1.75)
plt.title('sinogram')
plt.show()
#%%
# Detect stripes to calculate weights
stripe_weights = STRIPES_DETECT(proj3d,
                                vert_filter_size_perc=7,
                                radius_size=3)

plt.figure()
plt.imshow(stripe_weights[:,sliceno,:], cmap="gray", vmin= 0.0, vmax=0.6)
plt.title('stripes sinogram')
plt.show()
#%%
# Thresholding the obtained weight we can get a mask

stripesmask = STRIPES_MERGE(stripe_weights,
                             threshold=0.6,
                             stripe_length_perc=30.0,
                             stripe_depth_perc=100.0,
                             stripe_width_perc= 0.5,
                             sensitivity_perc=80.0)


plt.figure()
plt.imshow(stripesmask[:,sliceno,:], cmap="gray", vmin= 0.0, vmax=1)
plt.title('generated mask')
plt.show()
#%%
#remove data in proj3d where stripes are detected
proj3d[stripesmask == 1] = 0

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("___Inpainting in 2D using boundaries exatrapolation___")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
## plot 
fig = plt.figure()
plt.suptitle('Performance of ')
a=fig.add_subplot(1,2,1)
a.set_title('Missing data sinogram')
imgplot = plt.imshow(proj3d[:,sliceno,:],cmap="gray", vmin= 0.5, vmax=1.75)

# set parameters
pars = {'algorithm' : INPAINT_EUCL_WEIGHTED, 
        'input' : proj3d[:,sliceno,:],
        'maskData' : stripesmask[:,sliceno,:],
        'number_of_iterations' : 5,
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
imgplot = plt.imshow(inp_simple, cmap="gray", vmin= 0.5, vmax=1.75)
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
imgplot = plt.imshow(proj3d[:,sliceno,:],cmap="gray", vmin= 0.5, vmax=1.75)

# set parameters
pars = {'algorithm' : INPAINT_EUCL_WEIGHTED, 
        'input' : proj3d,
        'maskData' : stripesmask,
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
imgplot = plt.imshow(inp_simple[:,sliceno,:], cmap="gray", vmin= 0.5, vmax=1.75)
plt.title('{}'.format('Extrapolation inpainting results'))

#%%