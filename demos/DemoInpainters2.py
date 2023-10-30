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
    txt = r""
    for key, value in pars.items():
        if key == "algorithm":
            txt += "{0} = {1}".format(key, value.__name__)
        elif key == "input":
            txt += "{0} = {1}".format(key, np.shape(value))
        elif key == "maskData":
            txt += "{0} = {1}".format(key, np.shape(value))
        else:
            txt += "{0} = {1}".format(key, value)
        txt += "\n"
    return txt


###############################################################################
# 3d projection data
data = np.load("../data/proj3d_stripes.npz")
proj3d = data["proj3d"]


sliceno = 5
plt.figure()
plt.imshow(proj3d[:, sliceno, :], cmap="gray", vmin=0.5, vmax=1.75)
plt.title("sinogram")
plt.show()
# %%
# Detect stripes to calculate weights
stripe_weights = STRIPES_DETECT(proj3d, size=13, radius=5)


plt.figure()
plt.imshow(stripe_weights[:, sliceno, :], cmap="gray", vmin=0.0, vmax=0.6)
plt.title("stripes sinogram")
plt.show()
# %%
# Thresholding the obtained weight we can get a mask
stripesmask = STRIPES_MERGE(
    stripe_weights,
    threshold=0.6,
    min_stripe_length=300.0,
    min_stripe_depth=9,
    min_stripe_width=30,
    sensitivity_perc=80.0,
)


plt.figure()
plt.imshow(stripesmask[:, sliceno, :], cmap="gray", vmin=0.0, vmax=1)
plt.title("generated mask")
plt.show()

# remove data in proj3d where stripes are detected
proj3d[stripesmask == 1] = 0
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("___Inpainting in 2D using boundaries extrapolation___")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
## plot
fig = plt.figure()
plt.suptitle("Performance of ")
a = fig.add_subplot(1, 2, 1)
a.set_title("Missing data sinogram")
imgplot = plt.imshow(proj3d[:, sliceno, :], cmap="gray", vmin=0.5, vmax=1.75)

# set parameters
pars = {
    "algorithm": INPAINT_EUCL_WEIGHTED,
    "input": proj3d[:, sliceno, :],
    "maskData": stripesmask[:, sliceno, :],
    "number_of_iterations": 3,
    "windowsize_half": 5,
    "method_type": "random",
}

start_time = timeit.default_timer()
inpainted2d = INPAINT_EUCL_WEIGHTED(
    pars["input"],
    pars["maskData"],
    pars["number_of_iterations"],
    pars["windowsize_half"],
    pars["method_type"],
)

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ("elapsed time", timeit.default_timer() - start_time)
print(txtstr)
a = fig.add_subplot(1, 2, 2)
# these are matplotlib.patch.Patch properties
props = dict(boxstyle="round", facecolor="wheat", alpha=0.75)
# place a text box in upper left in axes coords
a.text(
    0.1,
    0.1,
    txtstr,
    transform=a.transAxes,
    fontsize=14,
    verticalalignment="top",
    bbox=props,
)
imgplot = plt.imshow(inpainted2d, cmap="gray", vmin=0.5, vmax=1.75)
plt.title("{}".format("2d inpainting results"))

# zoom on stripe
fig = plt.figure()
plt.rcParams.update({"font.size": 21})
plt.subplot(121)
plt.imshow(proj3d[586:800, sliceno, 200:380], vmin=0.4, vmax=1.75, cmap="gray")
plt.title("Sino+mask")
plt.subplot(122)
plt.imshow(inpainted2d[586:800, 200:380], cmap="gray", vmin=0.4, vmax=1.75)
plt.title("Random2d")
plt.show()
plt.savefig("inpainted2d_random.png", dpi=300)
# %%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("___Inpainting in 3D using boundaries extrapolation___")
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
        'number_of_iterations' : 3,
        'windowsize_half' : 5,
        'method_type' : 'random'}

start_time = timeit.default_timer()
inpainted3d = INPAINT_EUCL_WEIGHTED(pars['input'],
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
imgplot = plt.imshow(inpainted3d[:,sliceno,:], cmap="gray", vmin= 0.5, vmax=1.75)
plt.title('{}'.format('3d inpainting results'))

# NOTE: 3d version can bring the intensities from the incorrectly detected by mask regions to
# different Z-layers, so be more careful when working with it. The mask might needed to be expanded.
# It does, nevertheless, creates a more consistent in all 3 dimensions mask than 2D version which 
# might be a bit "jumpy"
    
# zoom on stripe
fig = plt.figure()
plt.rcParams.update({"font.size": 21})
plt.subplot(121)
plt.imshow(proj3d[586:800, sliceno, 200:380], vmin=0.4, vmax=1.75, cmap="gray")
plt.title("Sino+mask")
plt.subplot(122)
plt.imshow(inpainted3d[586:800, sliceno, 200:380], cmap="gray", vmin=0.4, vmax=1.75)
plt.title("Random3d")
plt.show()
#plt.savefig("inpainted3d_random.png", dpi=300)

# %%
