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
from skimage import filters,measure
import matplotlib.pyplot as plt
import napari
import mat73
from scipy.spatial import distance


# Convert to wavelength- thede are from the fits to the TS Bead Data. 
m=0.5
c=460 

# Read the files 

mat_fname=r"/Users/Mathew/Dropbox (Cambridge University)/Ed Code/LLPS analyisis/LifetimeImageData.mat"
mat_contents = sio.loadmat(mat_fname,squeeze_me=True)
lifetimes=mat_contents['lifetimeImageData']


mat_fname="/Users/Mathew/Dropbox (Cambridge University)/Ed Code/LLPS analyisis/LifetimeAlphaData.mat"
mat_contents2 = sio.loadmat(mat_fname,squeeze_me=True)
intensities=mat_contents2['lifetimeAlphaData']


# This is to make an summed intensity image over all wavelengths to perfom the thresholding on
sum_int=np.sum(intensities,axis=0)


# The below just thresholds the image based on intensity value - could also use Otsu method
thresh=0.1e+7
binary_im=sum_int>thresh

# Now get the stack only with the thresholded intensities or lifetimes present:
thresholded_intensities=binary_im*intensities
thresholded_lifetimes=binary_im*lifetimes


# This finds the wavelength at which the intensity is highest.
max_int=(np.argmax(thresholded_intensities,axis=0)*m+c)*binary_im

# Show plot- note the vmin and vmax. May need altering depending on the dataset. 
plt.imshow(max_int,cmap='rainbow',vmin=460,vmax=560)
plt.colorbar()


# Now to analyse some of the features.

labelled_image=measure.label(binary_im)
number_of_clusters=labelled_image.max()

# Make the arrays for the periphery vs. centre. There looks to be a difference between the outer and inner
# part of the droplets, hence need to separate.

periphery_image=np.zeros(labelled_image.shape)
centre_image=np.zeros(labelled_image.shape)

# Perform stats on the image:
measure_image=measure.regionprops_table(labelled_image,properties=('area','perimeter','centroid','orientation','major_axis_length','minor_axis_length'))
xcoord=measure_image["centroid-0"]
ycoord=measure_image["centroid-1"]
lengths=measure_image["major_axis_length"]  

# Go through each of clusters.
for num in range(1,number_of_clusters):
    
    # Ideally want to make a plot that shows distance from the centre point.
    distance_image=np.zeros(labelled_image.shape)
    
    # Select only the one droplet
    image_to_show=labelled_image==num
    
    # Make an image with just the coordinates
    wid=image_to_show.shape
    x = np.linspace(0, wid[0],wid[0])
    y = np.linspace(0, wid[1],wid[1])
    
    xv, yv = np.meshgrid(x, y)
    
    # Calculate the distances from the centre to each point in the droplet using the coordinate system. 
    image_dist=((yv-xcoord[num-1])**2+(xv-ycoord[num-1])**2)**(0.5)
    image_dist_clust=image_dist*image_to_show
 
    # This is the threshold that determines whether the pixel is in the periphery or the centre. 
    length=0.6*(lengths[num-1]/2)
    
    # Now make the image
    image_periphery=image_dist_clust>length
    image_centre=(image_dist_clust<=length)*image_to_show
    
    # Add to overall images that contain all of the clusters. 
    periphery_image=periphery_image+image_periphery
    centre_image=centre_image+image_centre

# Generate wavelength images
periphery_image_wl=periphery_image*max_int
centre_image_wl=centre_image*max_int 


# Show the plots. 
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].imshow(periphery_image)
axes[0].set_title("Periphery")  
axes[1].set_title("Centre")
axes[1].imshow(centre_image)


fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].imshow(periphery_image_wl,cmap='rainbow',vmin=460,vmax=560)
axes[0].set_title("Periphery")  
axes[1].set_title("Centre")
axes[1].imshow(centre_image_wl,cmap='rainbow',vmin=460,vmax=560)    

# Make histograms for periphery and centres. 
periph=periphery_image_wl.flatten()
cents=centre_image_wl.flatten()


fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].hist(periph, bins = 20,range=[450,550], rwidth=0.9,color='#0000ff')
axes[1].hist(cents, bins = 20,range=[450,550], rwidth=0.9,color='#ff0000')
axes[0].set_title("Periphery")  
axes[1].set_title("Centre")


