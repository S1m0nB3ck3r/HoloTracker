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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import trackpy as tp
import trackpy.diag as dg
import utils as utils
import cv2

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
        display_images = False

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

        if display_images:
            display(d_holo)

        # volume propagation by angular spectrum method
        t1 = time.perf_counter()
        propag.volume_propag_angular_spectrum_to_module(d_holo, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_volume_module,
            medium_wavelenght, cam_magnification, cam_pix_size, cam_nb_pix_X, cam_nb_pix_Y, 0.0, dz * 1e-6, nb_plane, 15, 125)
        t2 = time.perf_counter()

        if display_images:
            display(d_volume_module.max(axis=0))
            display(d_volume_module.max(axis=1))
            display(d_volume_module.max(axis=2))

        # focus on the volume (INPLACE)
        # focus.focus(d_volume_module, d_volume_module, focus_smooth_size, Focus_type.TENEGRAD)
        t3 = time.perf_counter()

        if display_images:
            display(d_volume_module.max(axis=0))
            display(d_volume_module.max(axis=1))
            display(d_volume_module.max(axis=2))

        #Projection and image centroid analysis
        d_focus_projection = d_volume_module.max(axis=0)

        d_focus_projection -= d_focus_projection.min()
        d_focus_projection /= d_focus_projection.max()
        d_focus_projection *= 255
        d_focus_projection_uint8  = d_focus_projection.astype(cp.uint8)
    
        h_focus_projection = cp.asnumpy(d_focus_projection_uint8)

        mean_val = np.mean(h_focus_projection)
        std_val = np.std(h_focus_projection)
        k = 7  # facteur multiplicateur

        threshold = mean_val + k * std_val
        _, binary = cv2.threshold(h_focus_projection, threshold, 255, cv2.THRESH_BINARY)
        # label + barycenters
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

        display_images = True

        if display_images:
            img_color = cv2.cvtColor(h_focus_projection, cv2.COLOR_GRAY2BGR)
            for i, (x, y) in enumerate(centroids):
                if i == 0:
                    continue  # ignorer le fond
                cv2.drawMarker(img_color, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
                cv2.putText(img_color, f"{i}", (int(x)+5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
            plt.figure(figsize=(10,5))
            plt.subplot(1,2,1)
            plt.title('Image originale')
            plt.imshow(img_color, cmap='gray')
            plt.axis('off')

            plt.subplot(1,2,2)
            plt.title('Seuillage adaptatif (Gaussian)')
            plt.imshow(binary, cmap='gray')
            plt.axis('off')

            plt.show()

        #Search best focus for earch xy coordinate of the centroid
        features = []
        for i, (x, y) in enumerate(centroids):
            x_um = x * dx
            y_um = y * dy
            z_um = d_volume_module[:, int(y), int(x)].argmax().get() * dz
            features.append((i_image, x_um, y_um, z_um))  
        
        positions = pd.DataFrame(features, columns = ['i_image','X','Y','Z'])
        # positions.to_csv(result_filename, mode = 'a', index = False, header = False)

        display_images = True

        if display_images:

            # ----- Volume dimensions -----
            Zmax = d_volume_module.shape[0] * dz
            Ymax = d_volume_module.shape[2] * dy
            Xmax = d_volume_module.shape[1] * dx

            # ----- 3D particle positions -----
            X = positions['X']
            Y = positions['Y']
            Z = positions['Z']

            # Optional: limit number of particles for performance
            max_particles = 1000
            if len(X) > max_particles:
                idx = np.random.choice(len(X), size=max_particles, replace=False)
                X = X[idx]
                Y = Y[idx]
                Z = Z[idx]

            # ----- 3D scatter plot -----
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter3D(X, Y, Z, color='red', s=10, label='Particles')

            # ----- Labels, limits, view -----
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(0, Xmax)
            ax.set_ylim(0, Ymax)
            ax.set_zlim(0, Zmax)
            ax.view_init(elev=30, azim=135)
            ax.legend()
            plt.tight_layout()
            plt.show()


            










