#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 2020

Demo to show the capability of autocropping function. 
It works to crop 2D projedctions as well as full 3D volumes. 

@author: Daniil Kazantsev 
"""
import numpy as np
import matplotlib.pyplot as plt
from larix.methods.misc import AUTOCROP

#  Load the 2D projection data (i23 beamline, DLS)
sample_data = np.load("../data/data2D_to_crop.npy")

plt.figure(1)
plt.imshow(sample_data, vmin=0, vmax=1.5, cmap="gray")
plt.title("2D tomographic projection")
plt.show()

print("Runnning autocropping in 2D...")

pars = {
    "input_data": sample_data,  # input grayscale image
    "threshold": 0.05,  # threhsold to control cropping strength
    "margin_skip": 10,  # skip number of pixels around the image border
    "statbox_size": 20,  # the size of the box to collect background statistics (mean)
    "increase_crop": 20,
}  # increse crop values to ensure better cropping

cropped_indices = AUTOCROP(
    pars["input_data"],
    pars["threshold"],
    pars["margin_skip"],
    pars["statbox_size"],
    pars["increase_crop"],
)

cropped_im = sample_data[
    int(cropped_indices[2]) : int(cropped_indices[3]),
    int(cropped_indices[0]) : int(cropped_indices[1]),
]

plt.figure(2)
plt.imshow(cropped_im, vmin=0, vmax=1.5, cmap="gray")
plt.title("cropped 2D projection")
plt.show()
# %%
