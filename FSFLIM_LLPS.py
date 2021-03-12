#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:22:57 2021

@author: Mathew
"""

from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import napari
import mat73


# Read the files 

mat_fname=r"/Users/Mathew/Dropbox (Cambridge University)/Ed Code/LLPS analyisis/LifetimeImageData.mat"
mat_contents = sio.loadmat(mat_fname,squeeze_me=True)
lifetimes=mat_contents['lifetimeImageData']


mat_fname="/Users/Mathew/Dropbox (Cambridge University)/Ed Code/LLPS analyisis/LifetimeAlphaData.mat"
mat_contents2 = sio.loadmat(mat_fname,squeeze_me=True)
intensities=mat_contents2['lifetimeAlphaData']



def see(number):
    plane=thresholded_intensities[number]
  
    plt.imshow(plane)

    

sum_int=np.sum(intensities,axis=0)

plt.imshow(sum_int)
plt.colorbar()

# The below just thresholds the image based on intensity value
thresh=0.1e+7
binary_im=sum_int>thresh


# Now get the stack only with the thresholded intensities present:
thresholded_intensities=binary_im*intensities
thresholded_lifetimes=binary_im*lifetimes


# Convert to wavelength- thede are from the fits to the TS Bead Data. 
m=0.5
c=460 

# This finds the wavelength at which the intensity is highest.
max_int=(np.argmax(thresholded_intensities,axis=0)*m+c)*binary_im

# Show plot- note the vmin and vmax. May need altering depending on the dataset. 
plt.imshow(max_int,cmap='rainbow',vmin=460,vmax=560)
plt.colorbar()
