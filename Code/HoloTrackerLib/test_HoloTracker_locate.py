# -*- coding: utf-8 -*-

"""
Filename: main_holo_python.py

Description:
Script for testing or executing the holograms analysis (locating objects in 3d coordinates on holograms, and linking positions to determine objects trajectories) without Labview software interface.

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
import focus as focus
from focus import Focus_type
from CCL3D import *
import pyximport; pyximport.install()
from cupyx.scipy.fft import fftn as cpxfftn
from cupyx.scipy.fft import ifftn as icpxfftn
from cupy.fft import rfft2, fft2, ifft2, fftshift, ifftshift, fftn, ifftn
from typeHolo import *
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
import trackpy.diag as dg
import utils as utils

# Parameters
loaded_params = utils.load_parameters('parameters.json')
display_images = loaded_params['display_images']
holo_directory = loaded_params['holo_directory']

result_filename = loaded_params['result_filename']
type_image = loaded_params['image_type']

#hologram parameters
wavelegth = loaded_params['wavelength']  # in m
medium_index = loaded_params['medium_index']  # index of refraction of the medium
medium_wavelenght = wavelegth / medium_index  # in m
cam_magnification = loaded_params['cam_magnification']
cam_nb_pix_X = loaded_params['cam_nb_pix_X']
cam_nb_pix_Y = loaded_params['cam_nb_pix_Y']
nb_plane = loaded_params['nb_plane']
cam_pix_size = loaded_params['cam_pix_size']
dx = 1000000 * cam_pix_size / cam_magnification #in µm
dy = 1000000 * cam_pix_size / cam_magnification #in µm
dz = 1000000 * loaded_params['plane_step']# in µm


# threshold, focus and CCL parameters
focus_smooth_size = loaded_params['focus_smooth_size']
threshold_value = loaded_params['threshold_value']
type_Threshold = type_threshold.NB_STD_DEV if loaded_params['type_Threshold'] == 'NB_STDVAR' else type_threshold.THRESHOLD
n_connectivity = loaded_params['n_connectivity']
particle_filter_size = loaded_params['particle_filter_size']

# memory allocations
h_holo = np.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = np.float32)
d_holo = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.float32)
d_fft_holo = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_fft_holo_propag = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_holo_propag = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.float32)
d_KERNEL = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_FFT_KERNEL = cp.zeros(shape = (cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.complex64)
d_volume_module = cp.zeros(shape = (nb_plane, cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.float32)
d_bin_volume_focus = cp.zeros(shape = (nb_plane, cam_nb_pix_Y, cam_nb_pix_X), dtype = cp.dtype(bool))

# mean Holo removing
h_mean_holo = calc_holo_moyen(holo_directory, cam_nb_pix_X, cam_nb_pix_Y, 'bmp')
d_mean_holo = cp.asarray(h_mean_holo)
# img_mean_holo = Image.fromarray(h_mean_holo)
#img_mean_holo.show()

i_image = np.uint64(0)
images = [image for image in os.listdir(holo_directory) if (image.split('.')[-1].lower() == type_image.lower())]
nb_images = len(images)

if os.path.exists(result_filename):
    os.remove(result_filename)

# for each hologram in directory
for image in os.listdir(holo_directory):
    if (image.split('.')[-1].lower() == type_image.lower()):

        ini_time = time.perf_counter()
        i_image += 1
        h_holo = read_image(os.path.join(holo_directory,image), cam_nb_pix_X, cam_nb_pix_Y)
        if display_images:
            display(h_holo)

        #Remove mean image
        h_holo = h_holo / h_mean_holo
        min = h_holo.min()
        max = h_holo.max()
        img = Image.fromarray((h_holo - min) * 255 / (max - min))
        d_holo = cp.asarray(h_holo)

        # volume propagation by angular spectrum method
        t1 = time.perf_counter()
        propag.volume_propag_angular_spectrum_to_module(d_holo, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_volume_module,
            medium_wavelenght, cam_magnification, cam_pix_size, cam_nb_pix_X, cam_nb_pix_Y, 0.0, dz * 1e-6, nb_plane, 15, 125)
        t2 = time.perf_counter()

        if display_images:
            display(d_volume_module.sum(axis=0))
            display(d_volume_module.sum(axis=1))
            display(d_volume_module.sum(axis=2))

        # focus on the volume (INPLACE)
        focus.focus(d_volume_module, d_volume_module, focus_smooth_size, Focus_type.TENEGRAD)
        t3 = time.perf_counter()

        if display_images:
            display(d_volume_module.sum(axis=0))
            display(d_volume_module.sum(axis=1))
            display(d_volume_module.sum(axis=2))

        #CCL3D
        if i_image == 1 :
            if type_Threshold == type_threshold.NB_STD_DEV:
                threshold = calc_threshold(d_volume_module, threshold_value)
            else :
                threshold = threshold_value

        d_labels, number_of_labels = CCL3D(d_bin_volume_focus, d_volume_module, type_threshold.THRESHOLD, threshold, n_connectivity)
        t4 = time.perf_counter()
        print('number of objects located: ', number_of_labels)

        #label analysis
        features = np.ndarray(shape = (number_of_labels,), dtype = dobjet)
        features = CCA_CUDA_float(d_labels, d_volume_module, number_of_labels, i_image, cam_nb_pix_X, cam_nb_pix_Y, nb_plane, dx, dy, dz)

        # features_filtered = CCL_filter(features, 1, 0)
        t5 = time.perf_counter()

        positions = pd.DataFrame(features, columns = ['i_image','baryX','baryY','baryZ','nb_pix'])
        positions.to_csv(result_filename, mode = 'a', index = False, header = False)

        t6 = time.perf_counter()
        
        t_propag = t2 - t1
        t_focus = t3 - t2
        t_ccl = t4 - t3
        t_cca = t5 - t4
        t_barycenters = t6 - t5
        total_time = t6 - ini_time
    
        print('t propagation : ', t_propag)
        print('t focus : ', t_focus)
        print('t ccl : ', t_ccl)
        print('t cca : ', t_cca)
        print('t barycenters : ', t_barycenters)
        print('total iteration time: ', total_time, "\n")

        if display_images:

            h_intensite = cp.asnumpy(d_volume_module**2).reshape((cam_nb_pix_X * cam_nb_pix_X * nb_plane, ))
            plt.hist(h_intensite, bins = 1000)
            plt.axis()
            plt.yscale('log')
            plt.show()

            #affichage 3D
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            Z = positions['baryZ']
            Y = positions['baryY']
            X = positions['baryX']
            ax.scatter3D(X, Y, Z)

            plt.show()


        










