# -*- coding: utf-8 -*-

"""
Filename: HoloTracker_locate_labview_wrapper.py

Description:
wrapper for the HoloTracker_locate interface coded with Labview to call all the Python code.
Author: Simon BECKER
Date: 2024-07-09

License:
GNU General Public License v3.0

Copyright (C) [2024] Simon BECKER

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import cupy as cp
import numpy as np
from cupyx import jit
import time
import os
from PIL import Image
from traitement_holo import *
import propagation as propag
import focus 
from focus import Focus_type
import typeHolo
from CCL3D import *
import pyximport; pyximport.install()
import pandas as pd

flag_allocated = False
flag_volume_propagated = False

# info hologram
wavelenght = 0
optical_index = 0
image_size_X = 0
image_size_Y = 0
pixel_size = 0
magnification = 0

#analyse parameters
distance_ini = 0
nb_plane = 0
propag_step = 0
low_pass_filter = 0
high_pass_filter = 0

focus_type = Focus_type.TENEGRAD
sum_size = 0 

threshold_type = type_threshold.NB_STD_DEV
nbStdVarThreshold = 0

n_connectivity = 26
min_voxel = 0
max_voxel = 0

def LV_test():
    return "OK"

def test(holo_info):
    wavelenght, optical_index, image_size_X, image_size_Y, pixel_size, magnification = holo_info

    return("lambda = " + str(wavelenght))


def LV_set_parameters(holo_parameters, propag_parameters, filter_parameters, focus_parameters, CCL_parameters):
    global wavelenght 
    global optical_index 
    global image_size_X 
    global image_size_Y 
    global pixel_size 
    global magnification

    global distance_ini
    global nb_plane 
    global propag_step 
    global low_pass_filter
    global high_pass_filter 

    global focus_type
    global sum_size 

    global threshold_type
    global nbStdVarThreshold

    global n_connectivity
    global min_voxel 
    global max_voxel


    #analyse param is a Labview cluster = python tuple
    # -> unpacking tuple
    wavelenght, optical_index, image_size_X, image_size_Y, pixel_size, magnification = holo_parameters
    #propagation unpacking
    distance_ini, nb_plane, propag_step = propag_parameters
    #filter unpacking
    high_pass_filter, low_pass_filter = filter_parameters
    #focus unpacking
    focus_type, sum_size = focus_parameters
    #binarisation parameters unpacking
    #CCL parameters unpacking
    nbStdVarThreshold, n_connectivity, min_voxel, max_voxel = CCL_parameters

    return( 
    str(wavelenght) + " " +\
    str(optical_index) + " " +\
    str(image_size_X) + " " +\
    str(image_size_Y) + " " +\
    str(pixel_size) + " " +\
    str(magnification) + " " + \
    str(distance_ini) + " " +\
    "nb plane: " + str(nb_plane) + " " +\
    str(propag_step) + " " +\
    "F LOW: " +str(low_pass_filter) + " " +\
    "F HIGH: " +str(high_pass_filter) + " " +\
    str(focus_type) + " " +\
    str(sum_size) + " " +\
    str(threshold_type) + " " +\
    str(nbStdVarThreshold) + " " +\
    str(n_connectivity) + " " +\
    str(min_voxel) + " " +\
    str(max_voxel) + " "
    )

def LV_get_parameters():

    global wavelenght 
    global optical_index 
    global image_size_X 
    global image_size_Y 
    global pixel_size 
    global magnification

    global distance_ini
    global nb_plane 
    global propag_step 
    global low_pass_filter
    global high_pass_filter 

    global focus_type
    global sum_size 

    global threshold_type
    global nbStdVarThreshold

    global n_connectivity
    global min_voxel 
    global max_voxel

    #analyse param is a Labview cluster = python tuple
    holo_parameters = wavelenght, optical_index, image_size_X, image_size_Y, pixel_size, magnification
    propag_parameters = distance_ini, nb_plane, propag_step
    filter_parameters = high_pass_filter, low_pass_filter
    focus_parameters = focus_type, sum_size
    CCL_parameters = nbStdVarThreshold, n_connectivity, min_voxel, max_voxel

    return (holo_parameters, propag_parameters, filter_parameters, focus_parameters, CCL_parameters)

def LV_allocate():

    global image_size_X 
    global image_size_Y 
    global nb_plane
    global flag_allocated

    #allocations
    global h_holo
    global h_mean_holo
    global d_holo
    global d_mean_holo
    global d_fft_holo
    global d_fft_holo_propag
    global d_holo_propag
    global d_KERNEL
    global d_FFT_KERNEL
    global d_volume_module
    global d_bin_volume_focus

    h_holo = np.zeros(shape = (image_size_Y, image_size_X), dtype = np.float32)
    h_mean_holo = np.zeros(shape = (image_size_Y, image_size_X), dtype = np.float32)
    d_holo = cp.zeros(shape = (image_size_Y, image_size_X), dtype = cp.float32)
    d_mean_holo = cp.zeros(shape = (image_size_Y, image_size_X), dtype = cp.float32)

    d_fft_holo = cp.zeros(shape = (image_size_Y, image_size_X), dtype = cp.complex64)
    d_fft_holo_propag = cp.zeros(shape = (image_size_Y, image_size_X), dtype = cp.complex64)
    d_holo_propag = cp.zeros(shape = (image_size_Y, image_size_X), dtype = cp.float32)
    d_KERNEL = cp.zeros(shape = (image_size_Y, image_size_X), dtype = cp.complex64)
    d_FFT_KERNEL = cp.zeros(shape = (image_size_Y, image_size_X), dtype = cp.complex64)
    d_volume_module = cp.zeros(shape = (nb_plane, image_size_Y, image_size_X), dtype = cp.float32)
    d_bin_volume_focus = cp.zeros(shape = (nb_plane, image_size_Y, image_size_X), dtype = cp.dtype(bool))

    flag_allocated = True

    return "allocation done"

def LV_open_hologram_file(path_holo, image_size_X, image_size_Y):

    return read_image(path_holo, image_size_X, image_size_Y)

def LV_set_hologram(np_holo):

    global flag_allocated
    global image_size_X 
    global image_size_Y 
    global h_holo
    global d_holo

    if flag_allocated:

        h_holo = np_holo
        d_holo = cp.asarray(h_holo)

        return "set hologram done"
    else:
        return "error, GPU not allocated"
    
def LV_get_hologram():

    global h_holo
    return h_holo
    

def LV_compute_mean_hologram(directoy_holograms, image_extension, image_size_X, image_size_Y):

    global h_mean_holo
    global d_mean_holo

    h_mean_holo = calc_holo_moyen(directoy_holograms, image_size_X, image_size_Y, image_extension)
    d_mean_holo = cp.asarray(h_mean_holo)
    
    return h_mean_holo

def LV_get_all_image_in_directory(dirPath, extension):

    files = os.listdir(dirPath) 
    images_files = []

    return list(filter(lambda image: image.split('.')[-1].lower() == extension.lower(), files))
    


def LV_volume_propagation():

    global wavelenght 
    global optical_index 
    global image_size_X 
    global image_size_Y 
    global pixel_size 
    global magnification
    global distance_ini
    global nb_plane 
    global propag_step 
    global low_pass_filter
    global high_pass_filter 
    global focus_type
    global sum_size 
    global threshold_type
    global nbStdVarThreshold
    global n_connectivity
    global min_voxel 
    global max_voxel
    global h_holo
    global d_holo
    global d_fft_holo
    global d_fft_holo_propag
    global d_holo_propag
    global d_KERNEL
    global d_FFT_KERNEL
    global d_volume_module
    global d_bin_volume_focus

    lamda_medium = wavelenght / optical_index

    if flag_allocated:
        if focus_type == 0:
            foc = Focus_type.SUM_OF_INTENSITY
            s_foc = "sum of intensity"
        elif focus_type == 1:
            foc = Focus_type.SUM_OF_LAPLACIAN
            s_foc = "sum of laplacian"
        elif focus_type == 2:
            foc = Focus_type.SUM_OF_VARIANCE
            s_foc = "sum of variance"
        elif focus_type == 3:
            foc = Focus_type.TENEGRAD
            s_foc = "TENENGRAD"

        propag.volume_propag_angular_spectrum_to_module(d_holo, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_volume_module,
            lamda_medium, magnification, pixel_size, image_size_X, image_size_Y, distance_ini, propag_step, nb_plane, int(high_pass_filter),int(low_pass_filter))
        
        focus.focus(d_volume_module, d_volume_module, sum_size, foc)
        flag_volume_propagated = True

        sizeZ, sizeY, sizeX,  = cp.shape(d_volume_module)

        return("propagation done " + s_foc + " sizeX :" + str(sizeX) + " sizeY :" + str(sizeY) + " sizeZ :" + str(sizeZ))
    else:
        return("GPU not allocated")
    
def LV_get_sum_XY():

    return(cp.asnumpy(d_volume_module.sum(axis = 0)))

def LV_get_max_XY():

    return(cp.asnumpy(cp.amax(d_volume_module,axis = 0)))

def LV_get_sum_XZ():

    return(cp.asnumpy(d_volume_module.sum(axis = 1)))

def LV_get_max_XZ():

    return(cp.asnumpy(cp.amax(d_volume_module,axis = 1)))

def LV_get_sum_YZ():

    return(cp.asnumpy(d_volume_module.sum(axis = 2)))

def LV_get_max_YZ():

    return(cp.asnumpy(cp.amax(d_volume_module,axis = 2)))

def LV_get_reconstruction_plane(plane):

    return(cp.asnumpy(d_volume_module[plane,:,:]))

def LV_get_binarised_plane(plane):

    return(cp.asnumpy(d_bin_volume_focus[plane,:,:]))

def LV_def_get_sub_volume(x, y, z, boxSizeXY, boxSizeZ):
    return get_sub_plane(x, y, z, boxSizeXY, boxSizeZ, d_volume_module)

    
def LV_get_mean_std_var_volume():

    global d_volume_module

    mean = cp.mean(d_volume_module)
    stdVar = cp.std(d_volume_module)

    #return(mean.item(), stdVar.item())
    return(mean.item(), stdVar.item())

def test_FFT():

    global d_holo
    global d_fft_holo

    d_FFT_HOLO = fftshift(fft2(d_holo, norm = 'ortho'))
    d_holo = cp.flip(cp.flip(fft2(fftshift(d_FFT_HOLO), norm = 'ortho'), axis=1), axis=0)

    return cp.asnumpy(cp.sqrt(cp.real(d_holo)**2 + cp.imag(d_holo)**2))

def get_AND_XY_bin_planes():
    return(cp.asnumpy(cp.clip(d_bin_volume_focus.sum(axis = 0), a_min = 0, a_max = 1)) * 255.0 )

def LV_find_features(threshold, image_number = 0):

    global wavelenght 
    global optical_index 
    global image_size_X 
    global image_size_Y 
    global pixel_size 
    global magnification
    global distance_ini
    global nb_plane 
    global propag_step 
    global low_pass_filter
    global high_pass_filter 
    global focus_type
    global sum_size 
    global nbStdVarThreshold
    global n_connectivity
    global min_voxel 
    global max_voxel
    global h_holo
    global d_holo
    global d_fft_holo
    global d_fft_holo_propag
    global d_holo_propag
    global d_KERNEL
    global d_FFT_KERNEL
    global d_volume_module
    global d_bin_volume_focus

    dx = pixel_size / magnification
    dy = pixel_size / magnification
    dz = propag_step

    if n_connectivity == 0 :
        connectivity = 6
    elif n_connectivity == 1:
        connectivity = 18
    else :
        connectivity = 26

    d_labels, number_of_labels = CCL3D(d_bin_volume_focus, d_volume_module, type_threshold.THRESHOLD, threshold, connectivity)

    #CCA(h_labels, h_focus_volume, features, i_image, dx, dy , dz)
    features_np = CCA_CUDA_float(d_labels, d_volume_module, number_of_labels, image_number, image_size_X, image_size_Y, nb_plane, dx, dy, dz)

    if (min_voxel != 0 or min_voxel != 1) and max_voxel != 0 :
        features_filtered = CCL_filter(features_np, min_voxel, max_voxel)
        return features_filtered
    else :
        return features_np

    # #changement de repere
    # features_np_2 = np.ndarray(shape = (number_of_labels,5), dtype = np.float32)

    # for i in range(number_of_labels):
    #     features_np_2[i, 0] = features_np[i, 0]
    #     features_np_2[i, 1] = features_np[i, 2]
    #     features_np_2[i, 2] = image_size_Y * dy - features_np[i, 1]  
    #     features_np_2[i, 3] = features_np[i, 3]
    #     features_np_2[i, 4] = features_np[i, 4]

    # return(features_np_2)


def LV_save_positions(path_file, features_np):
        
        features_dataframe = pd.DataFrame(features_np, columns = ['i_image','baryX','baryY','baryZ','nb_pix'])
        features_dataframe.to_csv(path_file, mode = 'a', index = False, header = False)
        return("save features done")



    