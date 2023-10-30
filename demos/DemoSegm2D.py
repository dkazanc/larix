#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:39:47 2020

Demo to show the capability of segmenting the phase of the 2D data with the
subsequent morphological processing

@author: Daniil Kazantsev
"""
import numpy as np
import matplotlib.pyplot as plt
from larix.methods.segmentation import REGION_GROW, MORPH_PROC_LINE

#  Load the 2D sample data (i23 beamline, DLS)
sample_data = np.load("../data/sample13076_2D.npy")

mask_init = np.uint8(np.zeros(np.shape(sample_data)))
mask_init[382:520, 217:330] = 1

fig = plt.figure()
plt.rcParams.update({"font.size": 21})
plt.subplot(121)
plt.imshow(sample_data, vmin=0, vmax=1, cmap="gray")
plt.title("Original Image")
plt.subplot(122)
plt.imshow(mask_init, vmin=0, vmax=1, cmap="gray")
plt.title("Phase specific initialised mask")
plt.show()
# %%
print("Runnning mask evolving (RegionGrow) segmentation in 2D...")

pars = {
    "input_data": sample_data,  # input grayscale image
    "maskData": mask_init,  # generated initialisation mask
    "threhsold": 5.0,  # threhsold controls where evolution stops (>=1)
    "iterationsNumb": 250,  # the number of iterations (depends on the size of the phase)
    "connectivity": 4,  # voxel connectivity rule, choose between 4 (2D), 6, 8 (2D), and 26
    "method": "mean",
}  # method to collect statistics from the mask (mean. median, value)

mask_evolved = REGION_GROW(
    pars["input_data"],
    pars["maskData"],
    pars["threhsold"],
    pars["iterationsNumb"],
    pars["connectivity"],
    pars["method"],
)

fig = plt.figure()
plt.rcParams.update({"font.size": 21})
plt.subplot(121)
plt.imshow(sample_data, vmin=0, vmax=1, cmap="gray")
plt.title("Original Image")
plt.subplot(122)
plt.imshow(mask_evolved, vmin=0, vmax=1, cmap="gray")
plt.title("Evolution of the mask")
plt.show()
# %%
print(
    "Morphological processing using line segments of the resulting 2D segmentation..."
)

pars = {
    "maskdata": mask_evolved,  # input binary mask
    "primeClass": 0,  # class to start morphological processing from
    "CorrectionWindow": 7,  # the non-local neighboorhood window
    "iterationsNumb": 15,
}  # iterations number

mask_morphed = MORPH_PROC_LINE(
    pars["maskdata"],
    pars["primeClass"],
    pars["CorrectionWindow"],
    pars["iterationsNumb"],
)

fig = plt.figure()
plt.rcParams.update({"font.size": 21})
plt.subplot(121)
plt.imshow(mask_evolved, vmin=0, vmax=1, cmap="gray")
plt.title("Evolved mask")
plt.subplot(122)
plt.imshow(mask_morphed, vmin=0, vmax=1, cmap="gray")
plt.title("Processed mask")
plt.show()
# %%
# save images and superimpose
from PIL import Image
import matplotlib

matplotlib.image.imsave("background.png", sample_data)
matplotlib.image.imsave("foreground.png", mask_morphed)

background = Image.open("background.png")
overlay = Image.open("foreground.png")

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.4)
new_img.save("new.png", "PNG")
# %%
