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
import focus 
from focus import Focus_type
import typeHolo
from CCL3D import *
import pyximport; pyximport.install()
#from CCanalyse import *
#from cupyx.scipy.fft import fftn as cpxfftn
#from cupyx.scipy.fft import ifftn as icpxfftn



from cupy.fft import rfft2, fft2, ifft2, fftshift, ifftshift, fftn, ifftn

from typeHolo import *
import math
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
import trackpy.diag as dg

# repertoire courant
path = r'C:\TRAVAIL\RepositoriesGithub\HoloTracker\Images_test_2'
result_filename = 'result_python_sum15_TENEGRAD_STD15_each.csv'
type_image = 'bmp'

nb_plan = 200

infoHolo = info_Holo()
infoHolo.lambdaMilieu = 660e-9 / 1.33
infoHolo.magnification = 40.0
infoHolo.nb_pix_X = 1024
infoHolo.nb_pix_Y = 512
infoHolo.pixSize = 7e-6

sizeX = infoHolo.nb_pix_X
sizeY = infoHolo.nb_pix_Y
sumSize = 15
#pas en µm
dx = 1000000 * infoHolo.pixSize / infoHolo.magnification
dy = 1000000 * infoHolo.pixSize / infoHolo.magnification
dz = 0.2

#seuillage, binarisation, et CCL
nbStdVarThreshold = 15
typeThreshold = type_threshold.THRESHOLD
n_connectivity = 26
filter_size = 0


#allocations
h_holo = np.zeros(shape = (sizeY, sizeX), dtype = np.float32)
d_holo = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)
d_fft_holo = cp.zeros(shape = (sizeY, sizeX), dtype = cp.complex64)
d_fft_holo_propag = cp.zeros(shape = (sizeY, sizeX), dtype = cp.complex64)
d_holo_propag = cp.zeros(shape = (sizeY, sizeX), dtype = cp.float32)
d_KERNEL = cp.zeros(shape = (sizeY, sizeX), dtype = cp.complex64)
d_FFT_KERNEL = cp.zeros(shape = (sizeY, sizeX), dtype = cp.complex64)
d_volume_module = cp.zeros(shape = (nb_plan, sizeY, sizeX), dtype = cp.float32)
d_bin_volume_focus = cp.zeros(shape = (nb_plan, sizeY, sizeX), dtype = cp.dtype(bool))

#calcul holo moyen
h_mean_holo = calc_holo_moyen(path, sizeX, sizeY, 'bmp')
min = h_mean_holo.min()
max = h_mean_holo.max()
mean = h_mean_holo.mean()
std_var = h_mean_holo.std()

d_mean_holo = cp.asarray(h_mean_holo)
img_mean_holo = Image.fromarray(h_mean_holo)
#img_mean_holo.show()

i_image = np.uint64(0)
images = [image for image in os.listdir(path) if (image.split('.')[-1].lower() == type_image.lower())]
nb_images = len(images)

if os.path.exists(result_filename):
    os.remove(result_filename)

#pour chaque hologramme du répertoire
for image in os.listdir(path):
    if (image.split('.')[-1].lower() == type_image.lower()):

        ini_time = time.perf_counter()
        i_image += 1

        #read image
        h_holo = read_image(os.path.join(path,image), sizeX, sizeY)

        # affichage(h_holo)

        #div image moyenne
        h_holo = h_holo / h_mean_holo
        min = h_holo.min()
        max = h_holo.max()
        img = Image.fromarray((h_holo - min) * 255 / (max - min))
        # img.show()

        #copie holo host vers gpu
        d_holo = cp.asarray(h_holo)

        t1 = time.perf_counter()

        #calcul du volume propagé SPECTRE ANGULAIRE
        propag.volume_propag_angular_spectrum_to_module(d_holo, d_fft_holo, d_KERNEL, d_fft_holo_propag, d_volume_module,
            infoHolo.lambdaMilieu, infoHolo.magnification, infoHolo.pixSize, infoHolo.nb_pix_X, infoHolo.nb_pix_Y, 0.0, dz * 1e-6, nb_plan, 15,125)
        
        t2 = time.perf_counter()

        #calcul du focus sur tout le volume
        focus.focus(d_volume_module, d_volume_module, sumSize, Focus_type.SUM_OF_LAPLACIAN)
        t3 = time.perf_counter()


        #CCL3D
        if i_image == 1:
            threshold = calc_threshold(d_volume_module, nbStdVarThreshold)

        sizeMeanXY = 100

        sizeMeanZ = 5

        t_s = time.perf_counter()
        d_labels, number_of_labels = CCL3D(d_bin_volume_focus, d_volume_module, typeThreshold, threshold, n_connectivity)
        t_f = time.perf_counter()
        print('nombre d\'objet trouvés: ', number_of_labels)
        print('t new_ccl : ', t_f - t_s)


        #print("nb pix: ", d_bin_volume_focus.sum())
        t4 = time.perf_counter()

        #analyse des labels
        features = np.ndarray(shape = (number_of_labels,), dtype = dobjet)
        print('nombre d\'objet trouvés: ', number_of_labels)

        features = CCA_CUDA_float(d_labels, d_volume_module, number_of_labels, i_image, sizeX, sizeY, nb_plan, dx, dy, dz)

        # features_filtered = CCL_filter(features, 1, 0)

        end_CCL_CCA_time = time.perf_counter()

        positions = pd.DataFrame(features, columns = ['i_image','baryX','baryY','baryZ','nb_pix'])

        positions.to_csv(result_filename, mode = 'a', index = False, header = False)

        t5 = time.perf_counter()
        
        final_time = time.perf_counter()
        t_propag = t2 - t1
        t_focus = t3 - t2
        t_ccl = t4 - t3
        t_cca = t5 - t4
        print('t propag : ', t_propag)
        print('t focus : ', t_focus)
        print('t ccl : ', t_ccl)
        print('t cca : ', t_cca, '\n')
        print('temps traitement: ', final_time - ini_time)


        # h_intensite = cp.asnumpy(d_volume_module**2).reshape((sizeX * sizeY * nb_plan, ))
        # plt.hist(h_intensite, bins = 1000)
        # plt.axis()
        # plt.yscale('log')
        # plt.show()

        # #affichage 3D
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # Z = positions['baryZ']
        # Y = positions['baryY']
        # X = positions['baryX']
        # ax.scatter3D(X, Y, Z)

        # plt.show()


        










