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
import time
import os
from PIL import Image
from HoloTrackerLib.traitement_holo import *
from HoloTrackerLib import propagation as propag
from HoloTrackerLib import focus as focus
from HoloTrackerLib.focus import Focus_type
from HoloTrackerLib.CCL3D import *
from HoloTrackerLib.typeHolo import *
from HoloTrackerLib import utils
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Load parameters
loaded_params = utils.load_parameters('parameters.json')
display_images = loaded_params['display_images']
holo_directory = loaded_params['holo_directory']
result_filename = loaded_params['result_filename']
type_image = loaded_params['image_type']

wavelegth = loaded_params['wavelength']
medium_index = loaded_params['medium_index']
medium_wavelenght = wavelegth / medium_index
cam_magnification = loaded_params['cam_magnification']
cam_nb_pix_X = loaded_params['cam_nb_pix_X']
cam_nb_pix_Y = loaded_params['cam_nb_pix_Y']
nb_plane = loaded_params['nb_plane']
cam_pix_size = loaded_params['cam_pix_size']
dx = 1e6 * cam_pix_size / cam_magnification
dy = 1e6 * cam_pix_size / cam_magnification
dz = 1e6 * loaded_params['plane_step']

# Memory allocations
d_volume_module = cp.zeros((nb_plane, cam_nb_pix_Y, cam_nb_pix_X), dtype=cp.float32)

# Mean image subtraction
h_mean_holo = calc_holo_moyen(holo_directory, cam_nb_pix_X, cam_nb_pix_Y, 'bmp')
d_mean_holo = cp.asarray(h_mean_holo)

# Iterate over holograms
images = [img for img in os.listdir(holo_directory) if img.lower().endswith(type_image.lower())]
i_image = 0

for image in images:

    display_images = False

    i_image += 1
    h_holo = read_image(os.path.join(holo_directory, image), cam_nb_pix_X, cam_nb_pix_Y)
    h_holo = h_holo / h_mean_holo
    d_holo = cp.asarray(h_holo)

    # Propagation
    propag.volume_propag_angular_spectrum_to_module(
        d_holo, cp.zeros_like(d_holo, dtype=cp.complex64),
        cp.zeros_like(d_holo, dtype=cp.complex64), cp.zeros_like(d_holo, dtype=cp.complex64),
        d_volume_module,
        medium_wavelenght, cam_magnification, cam_pix_size,
        cam_nb_pix_X, cam_nb_pix_Y, 0.0, dz * 1e-6, nb_plane, 15, 125
    )

    # Projection
    d_focus_projection = d_volume_module.max(axis=0)
    d_focus_projection -= d_focus_projection.min()
    d_focus_projection /= d_focus_projection.max()
    d_focus_projection *= 255
    h_focus_projection = cp.asnumpy(d_focus_projection.astype(cp.uint8))

    # Threshold
    k = 5   # Number of standard deviations for thresholding
    mean_val, std_val = np.mean(h_focus_projection), np.std(h_focus_projection)
    threshold = mean_val + k * std_val
    _, binary = cv2.threshold(h_focus_projection, threshold, 255, cv2.THRESH_BINARY)

    # Contour filtering
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accepted_centroids, rejected_centroids = [], []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
            if 0.3 <= circularity <= 1.2 and 20 < area < 1000:
                accepted_centroids.append((cx, cy))
            else:
                rejected_centroids.append((cx, cy))

    display_images = True

    # Display results
    if display_images:
        img_color = cv2.cvtColor(h_focus_projection, cv2.COLOR_GRAY2BGR)
        for i, (cx, cy) in enumerate(accepted_centroids):
            cv2.drawMarker(img_color, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
            cv2.putText(img_color, f"A{i}", (int(cx)+5, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        for i, (cx, cy) in enumerate(rejected_centroids):
            cv2.drawMarker(img_color, (int(cx), int(cy)), (255, 200, 100), cv2.MARKER_CROSS, 10, 1)
            cv2.putText(img_color, f"R{i}", (int(cx)+5, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,200,100), 1)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Accepted (Green) / Rejected (Blue)")
        plt.imshow(img_color)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Binary Threshold")
        plt.imshow(binary, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Z estimation from volume for accepted centroids
    features = []
    for x, y in accepted_centroids:
        x_um, y_um = x * dx, y * dy
        z_um = d_volume_module[:, int(y), int(x)].argmax().get() * dz
        features.append((i_image, x_um, y_um, z_um))

    positions = pd.DataFrame(features, columns=['i_image', 'X', 'Y', 'Z', 'nb_pix'])

    # Optional display of 3D scatter
    if display_images:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(positions['X'], positions['Y'], positions['Z'], color='red', s=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, cam_nb_pix_X * dx)
        ax.set_ylim(0, cam_nb_pix_Y * dy)
        ax.set_zlim(0, nb_plane * dz)
        ax.view_init(elev=30, azim=135)
        plt.tight_layout()
        plt.show()
